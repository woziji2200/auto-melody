"""Read and write Malody ``.mc`` charts stored in ``.mcz`` archives.

The implementation deliberately uses only the Python standard library.  Audio
decoding belongs to the model input pipeline; this module returns the exact
archive member and bytes so a decoder can consume or extract them as needed.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path, PurePosixPath
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import zipfile


AUDIO_SUFFIXES = frozenset({".ogg", ".wav", ".mp3", ".flac", ".m4a", ".aac"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
Beat = tuple[int, int, int]


class MalodyFormatError(ValueError):
    """Raised when a chart cannot be interpreted without guessing."""


def parse_beat(value: object, *, field: str = "beat") -> Beat:
    """Validate a Malody beat triplet and preserve its original representation.

    A beat ``[index, numerator, denominator]`` has the value
    ``index + numerator / denominator`` in BPM beat units.
    Malody accepts non-normalized fractions such as ``[17, 1, 1]``.
    """

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise MalodyFormatError(f"{field} must be a three-element array: {value!r}")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise MalodyFormatError(f"{field} elements must be integers: {value!r}")
    index, numerator, denominator = value
    if denominator <= 0:
        raise MalodyFormatError(f"{field} denominator must be positive: {value!r}")
    return index, numerator, denominator


def beat_fraction(value: Sequence[int]) -> Fraction:
    """Return a beat triplet as an exact number of Malody beat units."""

    index, numerator, denominator = parse_beat(value)
    return Fraction(index) + Fraction(numerator, denominator)


def fraction_to_beat(value: Fraction) -> Beat:
    """Return a normalized beat triplet for an exact Malody beat position."""

    bar = value.numerator // value.denominator
    remainder = value - bar
    return bar, remainder.numerator, remainder.denominator


def seconds_to_beat(seconds: float, *, bpm: float = 60.0, resolution: int = 1000) -> Beat:
    """Quantize seconds to a beat at a constant BPM.

    At the default 60 BPM one Malody beat is exactly one second, so a
    denominator of 1000 gives millisecond precision.  This is useful when a
    model predicts absolute timestamps but no musically meaningful beat grid.
    """

    if not math.isfinite(seconds):
        raise ValueError("seconds must be finite")
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be finite and positive")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    ticks = round(seconds * bpm * resolution / 60.0)
    bar, numerator = divmod(ticks, resolution)
    return fraction_to_beat(Fraction(bar) + Fraction(numerator, resolution))


@dataclass(frozen=True)
class TimingPoint:
    beat: Beat
    bpm: float
    seconds: float


class TimingMap:
    """Piecewise-constant BPM map with exact beat positions."""

    def __init__(self, timing_events: Sequence[Mapping[str, Any]]) -> None:
        if not timing_events:
            raise MalodyFormatError("chart has no BPM timing points")

        # At duplicate positions Malody effectively has a zero-length segment.
        # Keeping the final entry is deterministic and matches event ordering.
        deduplicated: dict[Fraction, tuple[Beat, float]] = {}
        for index, event in enumerate(timing_events):
            if not isinstance(event, Mapping):
                raise MalodyFormatError(f"time[{index}] must be an object")
            beat = parse_beat(event.get("beat"), field=f"time[{index}].beat")
            bpm_value = event.get("bpm")
            if (
                not isinstance(bpm_value, (int, float))
                or isinstance(bpm_value, bool)
                or not math.isfinite(float(bpm_value))
                or float(bpm_value) <= 0
            ):
                raise MalodyFormatError(f"time[{index}].bpm must be positive: {bpm_value!r}")
            deduplicated[beat_fraction(beat)] = (beat, float(bpm_value))

        ordered = sorted(deduplicated.items())
        self._positions = [position for position, _ in ordered]
        self._beats = [item[0] for _, item in ordered]
        self._bpms = [item[1] for _, item in ordered]

        # Build relative anchors, then shift them so chart beat zero is t=0.
        anchors = [0.0]
        for index in range(1, len(self._positions)):
            distance = self._positions[index] - self._positions[index - 1]
            anchors.append(anchors[-1] + float(distance) * 60.0 / self._bpms[index - 1])
        self._seconds = anchors
        zero_time = self._beat_fraction_to_seconds(Fraction(0))
        self._seconds = [value - zero_time for value in self._seconds]

    @property
    def points(self) -> tuple[TimingPoint, ...]:
        return tuple(
            TimingPoint(beat=beat, bpm=bpm, seconds=seconds)
            for beat, bpm, seconds in zip(self._beats, self._bpms, self._seconds)
        )

    def _beat_fraction_to_seconds(self, position: Fraction) -> float:
        index = bisect_right(self._positions, position) - 1
        if index < 0:
            index = 0
        distance = position - self._positions[index]
        return self._seconds[index] + float(distance) * 60.0 / self._bpms[index]

    def beat_to_seconds(self, beat: Sequence[int]) -> float:
        """Convert a Malody beat to seconds on the chart timeline."""

        return self._beat_fraction_to_seconds(beat_fraction(beat))

    def seconds_to_beat(self, seconds: float) -> Beat:
        """Convert chart-timeline seconds back to a normalized beat triplet."""

        if not math.isfinite(seconds):
            raise ValueError("seconds must be finite")
        index = bisect_right(self._seconds, seconds) - 1
        if index < 0:
            index = 0
        position = self._positions[index] + Fraction(
            str((seconds - self._seconds[index]) * self._bpms[index] / 60.0)
        )
        return fraction_to_beat(position.limit_denominator(1_000_000))


@dataclass(frozen=True)
class PlayableNote:
    beat: Beat
    column: int
    end_beat: Beat | None = None

    @property
    def is_hold(self) -> bool:
        return self.end_beat is not None


@dataclass(frozen=True)
class TimedNote:
    """A 4K note aligned to time zero of its referenced audio file."""

    start_seconds: float
    column: int
    end_seconds: float | None = None

    @property
    def is_hold(self) -> bool:
        return self.end_seconds is not None


@dataclass(frozen=True)
class AudioEvent:
    beat: Beat
    sound: str
    offset_ms: int
    event_type: int | None
    volume: int | float | None
    column: int | None
    member: str | None
    member_size: int | None
    member_crc32: int | None

    @property
    def is_autoplay(self) -> bool:
        return self.event_type == 1 and self.column is None


@dataclass
class LoadedChart:
    archive_path: Path
    member: str
    data: dict[str, Any]
    timeline: TimingMap
    notes: tuple[PlayableNote, ...]
    sound_events: tuple[AudioEvent, ...]

    @property
    def meta(self) -> Mapping[str, Any]:
        return self.data["meta"]

    @property
    def mode(self) -> int | None:
        value = self.meta.get("mode")
        return value if isinstance(value, int) else None

    @property
    def columns(self) -> int | None:
        mode_ext = self.meta.get("mode_ext")
        if not isinstance(mode_ext, Mapping):
            return None
        value = mode_ext.get("column")
        return value if isinstance(value, int) else None

    @property
    def is_4k(self) -> bool:
        return self.mode == 0 and self.columns == 4

    @property
    def resolved_audio_events(self) -> tuple[AudioEvent, ...]:
        """Return resolved events, collapsing byte-for-byte duplicate events."""

        result: list[AudioEvent] = []
        seen: set[tuple[Any, ...]] = set()
        for event in self.sound_events:
            if event.member is None:
                continue
            signature = (
                beat_fraction(event.beat),
                event.member,
                event.offset_ms,
                event.event_type,
                event.volume,
                event.column,
            )
            if signature not in seen:
                seen.add(signature)
                result.append(event)
        return tuple(result)

    @property
    def audio_status(self) -> str:
        """Classify whether the chart can use one existing full-track audio."""

        events = self.resolved_audio_events
        if len(events) == 1 and events[0].is_autoplay:
            return "ready"
        if not events:
            return "missing"
        return "composite"

    @property
    def primary_audio_event(self) -> AudioEvent:
        if self.audio_status != "ready":
            raise MalodyFormatError(
                f"{self.archive_path}:{self.member} has audio status {self.audio_status!r}, "
                "so it has no unambiguous full-track audio"
            )
        return self.resolved_audio_events[0]

    @property
    def audio_start_chart_seconds(self) -> float:
        """Chart time at which byte zero of the primary audio starts playing."""

        event = self.primary_audio_event
        return self.timeline.beat_to_seconds(event.beat) + event.offset_ms / 1000.0

    def timed_notes(self) -> tuple[TimedNote, ...]:
        """Return notes in the primary audio file's time coordinate system."""

        origin = self.audio_start_chart_seconds
        result = []
        for note in self.notes:
            start = self.timeline.beat_to_seconds(note.beat) - origin
            end = None
            if note.end_beat is not None:
                end = self.timeline.beat_to_seconds(note.end_beat) - origin
            result.append(TimedNote(start_seconds=start, column=note.column, end_seconds=end))
        return tuple(result)

    def read_audio_bytes(self) -> bytes:
        """Read the primary audio member without extracting the whole archive."""

        member = self.primary_audio_event.member
        assert member is not None
        with zipfile.ZipFile(self.archive_path) as archive:
            return archive.read(member)


def _normalized_member(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _audio_member_lookup(names: Sequence[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact: dict[str, str] = {}
    by_basename: dict[str, list[str]] = defaultdict(list)
    for original in names:
        normalized = _normalized_member(original)
        if PurePosixPath(normalized).suffix.lower() not in AUDIO_SUFFIXES:
            continue
        exact[normalized.lower()] = original
        by_basename[PurePosixPath(normalized).name.lower()].append(original)
    return exact, by_basename


def _resolve_audio_member(
    chart_member: str,
    sound: str,
    exact: Mapping[str, str],
    by_basename: Mapping[str, Sequence[str]],
) -> str | None:
    sound_path = PurePosixPath(_normalized_member(sound))
    chart_parent = PurePosixPath(_normalized_member(chart_member)).parent
    relative = _normalized_member(str(chart_parent / sound_path)).lower()
    if relative in exact:
        return exact[relative]
    direct = _normalized_member(str(sound_path)).lower()
    if direct in exact:
        return exact[direct]
    matches = by_basename.get(sound_path.name.lower(), ())
    return matches[0] if len(matches) == 1 else None


def _decode_chart(raw: bytes, *, source: str) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MalodyFormatError(f"invalid JSON in {source}: {error}") from error
    if not isinstance(data, dict):
        raise MalodyFormatError(f"chart root in {source} must be an object")
    if not isinstance(data.get("meta"), dict):
        raise MalodyFormatError(f"chart {source} has no meta object")
    for field in ("time", "note"):
        if not isinstance(data.get(field), list):
            raise MalodyFormatError(f"chart {source} has no {field} array")
    if "effect" in data and not isinstance(data["effect"], list):
        raise MalodyFormatError(f"chart {source} effect must be an array")
    return data


def _load_chart_from_archive(
    path: Path,
    archive: zipfile.ZipFile,
    member: str,
    *,
    audio_lookup: tuple[dict[str, str], dict[str, list[str]]] | None = None,
) -> LoadedChart:
    try:
        raw = archive.read(member)
    except KeyError as error:
        raise MalodyFormatError(f"chart member not found: {member}") from error
    data = _decode_chart(raw, source=f"{path}:{member}")
    timeline = TimingMap(data["time"])
    mode_ext = data["meta"].get("mode_ext")
    columns_value = mode_ext.get("column") if isinstance(mode_ext, Mapping) else None
    columns = columns_value if isinstance(columns_value, int) else None
    exact, by_basename = audio_lookup or _audio_member_lookup(archive.namelist())

    notes: list[PlayableNote] = []
    sound_events: list[AudioEvent] = []
    for index, event in enumerate(data["note"]):
        if not isinstance(event, Mapping):
            raise MalodyFormatError(f"note[{index}] must be an object")
        beat = parse_beat(event.get("beat"), field=f"note[{index}].beat")
        column = event.get("column")
        if column is not None:
            if not isinstance(column, int) or isinstance(column, bool):
                raise MalodyFormatError(f"note[{index}].column must be an integer")
            if columns is not None and not 0 <= column < columns:
                raise MalodyFormatError(
                    f"note[{index}].column {column} is outside a {columns}-column chart"
                )
            end_beat = None
            if "endbeat" in event:
                end_beat = parse_beat(event["endbeat"], field=f"note[{index}].endbeat")
                if beat_fraction(end_beat) <= beat_fraction(beat):
                    raise MalodyFormatError(f"note[{index}] hold ends before it starts")
            notes.append(PlayableNote(beat=beat, column=column, end_beat=end_beat))

        sound_value = event.get("sound")
        if isinstance(sound_value, str) and sound_value:
            offset_value = event.get("offset", 0)
            if not isinstance(offset_value, int) or isinstance(offset_value, bool):
                raise MalodyFormatError(f"note[{index}].offset must be an integer")
            event_type = event.get("type")
            resolved_member = _resolve_audio_member(member, sound_value, exact, by_basename)
            member_info = archive.getinfo(resolved_member) if resolved_member is not None else None
            sound_events.append(
                AudioEvent(
                    beat=beat,
                    sound=sound_value,
                    offset_ms=offset_value,
                    event_type=event_type if isinstance(event_type, int) else None,
                    volume=event.get("vol") if isinstance(event.get("vol"), (int, float)) else None,
                    column=column if isinstance(column, int) else None,
                    member=resolved_member,
                    member_size=member_info.file_size if member_info is not None else None,
                    member_crc32=member_info.CRC if member_info is not None else None,
                )
            )

    return LoadedChart(
        archive_path=path,
        member=member,
        data=data,
        timeline=timeline,
        notes=tuple(notes),
        sound_events=tuple(sound_events),
    )


def load_chart(path: str | Path, member: str) -> LoadedChart:
    """Load one ``.mc`` member from an ``.mcz`` archive."""

    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            return _load_chart_from_archive(archive_path, archive, member)
    except zipfile.BadZipFile as error:
        raise MalodyFormatError(f"not a valid MCZ/ZIP archive: {archive_path}") from error


def iter_archive_charts(path: str | Path) -> Iterator[LoadedChart]:
    """Yield every ``.mc`` chart in one MCZ archive."""

    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            lookup = _audio_member_lookup(archive.namelist())
            for member in archive.namelist():
                if member.lower().endswith(".mc"):
                    yield _load_chart_from_archive(
                        archive_path, archive, member, audio_lookup=lookup
                    )
    except zipfile.BadZipFile as error:
        raise MalodyFormatError(f"not a valid MCZ/ZIP archive: {archive_path}") from error


def iter_dataset_charts(dataset_dir: str | Path) -> Iterator[LoadedChart]:
    """Yield every parseable ``.mc`` chart below a dataset directory."""

    root = Path(dataset_dir)
    for path in sorted(root.rglob("*.mcz")):
        try:
            with zipfile.ZipFile(path) as archive:
                lookup = _audio_member_lookup(archive.namelist())
                for member in archive.namelist():
                    if member.lower().endswith(".mc"):
                        try:
                            yield _load_chart_from_archive(
                                path, archive, member, audio_lookup=lookup
                            )
                        except MalodyFormatError:
                            continue
        except (OSError, zipfile.BadZipFile):
            continue


def _manifest_record(chart: LoadedChart, root: Path) -> dict[str, Any]:
    meta = chart.meta
    song = meta.get("song") if isinstance(meta.get("song"), Mapping) else {}
    effects = chart.data.get("effect", [])
    warnings = []
    if any(isinstance(event, Mapping) and "jump" in event for event in effects):
        warnings.append("jump_effect")
    if len(chart.resolved_audio_events) != len(
        [event for event in chart.sound_events if event.member is not None]
    ):
        warnings.append("duplicate_audio_event")

    record: dict[str, Any] = {
        "archive": str(chart.archive_path.relative_to(root)),
        "chart_member": chart.member,
        "song_id": song.get("id"),
        "chart_id": meta.get("id"),
        "title": song.get("title"),
        "artist": song.get("artist"),
        "creator": meta.get("creator"),
        "version": meta.get("version"),
        "audio_status": chart.audio_status,
        "audio_member": None,
        "audio_offset_ms": None,
        "audio_event_beat": None,
        "audio_size_bytes": None,
        "audio_crc32": None,
        "audio_fingerprint": None,
        "note_count": len(chart.notes),
        "hold_count": sum(note.is_hold for note in chart.notes),
        "timing_point_count": len(chart.timeline.points),
        "unresolved_sound_event_count": sum(event.member is None for event in chart.sound_events),
        "warnings": warnings,
    }
    if chart.audio_status == "ready":
        event = chart.primary_audio_event
        record["audio_member"] = event.member
        record["audio_offset_ms"] = event.offset_ms
        record["audio_event_beat"] = list(event.beat)
        record["audio_size_bytes"] = event.member_size
        if event.member_crc32 is not None and event.member_size is not None:
            record["audio_crc32"] = f"{event.member_crc32:08x}"
            record["audio_fingerprint"] = f"{event.member_crc32:08x}:{event.member_size}"
        timed = chart.timed_notes()
        record["first_note_seconds"] = min(note.start_seconds for note in timed)
        record["last_note_seconds"] = max(
            note.end_seconds if note.end_seconds is not None else note.start_seconds
            for note in timed
        )
    return record


def scan_dataset(dataset_dir: str | Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Scan the dataset and return 4K manifest records plus status counts."""

    root = Path(dataset_dir).resolve()
    records = []
    status_counts: Counter[str] = Counter()
    for chart in iter_dataset_charts(root):
        if not chart.is_4k:
            continue
        record = _manifest_record(chart, root)
        records.append(record)
        status_counts[record["audio_status"]] += 1
    return records, status_counts


def write_manifest(dataset_dir: str | Path, output: str | Path) -> Counter[str]:
    """Write one JSON object per 4K chart and return audio status counts."""

    records, counts = scan_dataset(dataset_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return counts


def build_key_chart_from_seconds(
    *,
    audio_filename: str,
    notes: Iterable[Mapping[str, Any] | TimedNote],
    title: str,
    artist: str = "",
    creator: str = "auto-melody",
    version: str = "4K Generated",
    bpm: float = 60.0,
    resolution: int = 1000,
    song_id: int = 0,
    chart_id: int = 0,
    modified_time: int | None = None,
) -> dict[str, Any]:
    """Build a playable 4K MC object from model-predicted audio timestamps."""

    if PurePosixPath(audio_filename).suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"unsupported audio filename: {audio_filename}")
    output_notes: list[dict[str, Any]] = []
    previous_time = -math.inf
    for index, value in enumerate(notes):
        if isinstance(value, TimedNote):
            start = value.start_seconds
            end = value.end_seconds
            column = value.column
        else:
            start = value.get("time")
            end = value.get("end_time")
            column = value.get("column")
        if not isinstance(start, (int, float)) or not math.isfinite(float(start)) or start < 0:
            raise ValueError(f"notes[{index}].time must be a finite non-negative number")
        if not isinstance(column, int) or isinstance(column, bool) or not 0 <= column < 4:
            raise ValueError(f"notes[{index}].column must be in 0..3")
        if float(start) < previous_time:
            raise ValueError("notes must be sorted by time")
        event: dict[str, Any] = {
            "beat": list(seconds_to_beat(float(start), bpm=bpm, resolution=resolution)),
            "column": column,
        }
        if end is not None:
            if not isinstance(end, (int, float)) or not math.isfinite(float(end)) or end <= start:
                raise ValueError(f"notes[{index}].end_time must be later than time")
            end_beat = seconds_to_beat(float(end), bpm=bpm, resolution=resolution)
            if beat_fraction(end_beat) <= beat_fraction(event["beat"]):
                raise ValueError(
                    f"notes[{index}] becomes a zero-length hold at resolution {resolution}"
                )
            event["endbeat"] = list(end_beat)
        output_notes.append(event)
        previous_time = float(start)

    output_notes.append(
        {
            "beat": [0, 0, 1],
            "sound": PurePosixPath(audio_filename).name,
            "vol": 100,
            "offset": 0,
            "type": 1,
        }
    )
    return {
        "meta": {
            "$ver": 0,
            "creator": creator,
            "background": "",
            "version": version,
            "id": chart_id,
            "mode": 0,
            "time": int(time.time()) if modified_time is None else modified_time,
            "song": {"title": title, "artist": artist, "id": song_id},
            "mode_ext": {"column": 4},
        },
        "time": [{"beat": [0, 0, 1], "bpm": float(bpm)}],
        "effect": [],
        "note": output_notes,
    }


def pack_mcz(
    output: str | Path,
    *,
    chart: Mapping[str, Any],
    audio_path: str | Path,
    cover_path: str | Path | None = None,
    chart_name: str = "chart.mc",
) -> None:
    """Package a chart and its referenced audio as a Malody MCZ archive."""

    pack_mcz_charts(
        output,
        charts={chart_name: chart},
        audio_path=audio_path,
        cover_path=cover_path,
    )


def pack_mcz_charts(
    output: str | Path,
    *,
    charts: Mapping[str, Mapping[str, Any]],
    audio_path: str | Path,
    cover_path: str | Path | None = None,
) -> None:
    """Package multiple charts that share one referenced audio file."""

    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(audio)
    if audio.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"unsupported audio file: {audio}")
    cover = Path(cover_path) if cover_path is not None else None
    if cover is not None:
        if not cover.is_file():
            raise FileNotFoundError(cover)
        if cover.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported cover image: {cover}")
        if cover.name == audio.name:
            raise ValueError("cover and audio filenames must be different")
    if not charts:
        raise ValueError("at least one chart is required")
    encoded_charts: dict[str, bytes] = {}
    for chart_name, chart in charts.items():
        if PurePosixPath(chart_name).name != chart_name or not chart_name.lower().endswith(".mc"):
            raise ValueError("chart names must be plain .mc filenames")
        encoded = json.dumps(chart, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        decoded = _decode_chart(encoded, source=chart_name)
        if cover is not None and decoded["meta"].get("background") != cover.name:
            raise MalodyFormatError(
                f"{chart_name} does not reference the packaged cover filename {cover.name!r}"
            )
        referenced_audio = {
            PurePosixPath(str(event["sound"])).name
            for event in decoded["note"]
            if isinstance(event, Mapping) and isinstance(event.get("sound"), str)
        }
        if audio.name not in referenced_audio:
            raise MalodyFormatError(
                f"{chart_name} does not reference the packaged audio filename {audio.name!r}"
            )
        encoded_charts[chart_name] = encoded
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for chart_name, encoded in encoded_charts.items():
            archive.writestr(f"0/{chart_name}", encoded)
        archive.write(audio, f"0/{audio.name}")
        if cover is not None:
            archive.write(cover, f"0/{cover.name}")

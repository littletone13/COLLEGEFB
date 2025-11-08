"""Utilities for loading and resolving calibrated prop parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

CalibrationMode = Literal["auto", "heuristic", "calibrated"]


@dataclass(frozen=True)
class VolumeCalibration:
    phi: float
    variance_scale: float
    zero_inflation: Optional[float]
    mean_scale: float
    samples: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VolumeCalibration":
        return cls(
            phi=float(data.get("phi", 0.0)),
            variance_scale=float(data.get("variance_scale", 1.0)),
            zero_inflation=(
                float(data["zero_inflation"])
                if data.get("zero_inflation") is not None
                else None
            ),
            mean_scale=float(data.get("mean_scale", 1.0)),
            samples=int(data.get("samples", 0)),
        )


@dataclass(frozen=True)
class YardageCalibration:
    sigma: float
    variance_scale: float
    mean_scale: float
    samples: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "YardageCalibration":
        return cls(
            sigma=float(data.get("sigma", 0.8)),
            variance_scale=float(data.get("variance_scale", 1.0)),
            mean_scale=float(data.get("mean_scale", 1.0)),
            samples=int(data.get("samples", 0)),
        )


class PropCalibrations:
    """Container for calibrated parameters and helper accessors."""

    def __init__(
        self,
        *,
        volume: Dict[str, Dict[str, VolumeCalibration]] | None = None,
        yardage: Dict[str, Dict[str, YardageCalibration]] | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> None:
        self._volume: Dict[str, Dict[str, VolumeCalibration]] = {}
        if volume:
            for group, entries in volume.items():
                table: Dict[str, VolumeCalibration] = {}
                for key, value in entries.items():
                    norm_key = self._normalise_key(key)
                    table[norm_key] = value
                self._volume[group.lower()] = table
        self._yardage: Dict[str, Dict[str, YardageCalibration]] = {}
        if yardage:
            for stat, entries in yardage.items():
                table = {}
                for key, value in entries.items():
                    table[self._normalise_key(key)] = value
                self._yardage[stat.lower()] = table
        self.meta: Dict[str, Any] = meta or {}
        self.min_samples: int = int(self.meta.get("min_samples", 20))

    @staticmethod
    def _normalise_key(key: str) -> str:
        return key.strip().lower()

    def _resolve_table(self, group: str, *, is_volume: bool) -> Dict[str, Any]:
        collection = self._volume if is_volume else self._yardage
        return collection.get(group.lower(), {})

    def get_volume(
        self,
        group: str,
        keys: Sequence[str],
        *,
        mode: CalibrationMode = "auto",
    ) -> Optional[VolumeCalibration]:
        table = self._resolve_table(group, is_volume=True)
        if not table:
            return None
        for key in keys:
            norm_key = self._normalise_key(key)
            entry = table.get(norm_key)
            if entry is None:
                continue
            if mode == "heuristic":
                continue
            if mode == "calibrated" or entry.samples >= self.min_samples:
                return entry
        return None

    def get_yardage(
        self,
        stat: str,
        keys: Sequence[str],
        *,
        mode: CalibrationMode = "auto",
    ) -> Optional[YardageCalibration]:
        table = self._resolve_table(stat, is_volume=False)
        if not table:
            return None
        for key in keys:
            norm_key = self._normalise_key(key)
            entry = table.get(norm_key)
            if entry is None:
                continue
            if mode == "heuristic":
                continue
            if mode == "calibrated" or entry.samples >= self.min_samples:
                return entry
        return None


def _parse_volume(data: Dict[str, Any]) -> Dict[str, Dict[str, VolumeCalibration]]:
    out: Dict[str, Dict[str, VolumeCalibration]] = {}
    for group, entries in data.items():
        group_table: Dict[str, VolumeCalibration] = {}
        if isinstance(entries, dict):
            for key, entry in entries.items():
                if isinstance(entry, dict):
                    group_table[key.strip().lower()] = VolumeCalibration.from_dict(entry)
        out[group.lower()] = group_table
    return out


def _parse_yardage(data: Dict[str, Any]) -> Dict[str, Dict[str, YardageCalibration]]:
    out: Dict[str, Dict[str, YardageCalibration]] = {}
    for stat, entries in data.items():
        stat_table: Dict[str, YardageCalibration] = {}
        if isinstance(entries, dict):
            for key, entry in entries.items():
                if isinstance(entry, dict):
                    stat_table[key.strip().lower()] = YardageCalibration.from_dict(entry)
        out[stat.lower()] = stat_table
    return out


def load_calibrations(path: Path | str | None = None) -> PropCalibrations:
    """Load a calibration JSON/YAML file."""

    if path is None:
        path = default_calibration_path()
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    volume_data = _parse_volume(payload.get("volume", {}))
    yardage_data = _parse_yardage(payload.get("yardage", {}))
    meta = payload.get("meta", {})
    return PropCalibrations(volume=volume_data, yardage=yardage_data, meta=meta)


def default_calibration_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "player_prop_calibration.json"


_DEFAULT_CACHE: Optional[PropCalibrations] = None
_DEFAULT_ERROR: Optional[Exception] = None


def get_default_calibrations() -> Optional[PropCalibrations]:
    """Return cached default calibrations if available."""

    global _DEFAULT_CACHE, _DEFAULT_ERROR
    if _DEFAULT_CACHE is not None:
        return _DEFAULT_CACHE
    if _DEFAULT_ERROR is not None:
        return None
    try:
        _DEFAULT_CACHE = load_calibrations()
    except Exception as exc:  # pragma: no cover - defensive caching
        _DEFAULT_ERROR = exc
        return None
    return _DEFAULT_CACHE


__all__ = [
    "VolumeCalibration",
    "YardageCalibration",
    "PropCalibrations",
    "load_calibrations",
    "get_default_calibrations",
    "default_calibration_path",
]

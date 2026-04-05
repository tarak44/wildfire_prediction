from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from wildfire_mlops.constants import SUPPORTED_IMAGE_EXTS


@dataclass
class SyntheticWeatherRecord:
    temperature_c: float
    humidity_pct: float
    wind_speed_kph: float
    drought_index: float
    vegetation_dryness: float
    days_since_rain: float
    latitude: float | None
    longitude: float | None
    risk_score: float
    temporal_sequence: list[dict[str, float]]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_lat_lon(image_path: Path) -> tuple[float | None, float | None]:
    try:
        stem = image_path.stem
        lon_str, lat_str = stem.split(",", maxsplit=1)
        return float(lat_str), float(lon_str)
    except Exception:
        return None, None


def _seasonal_offset(latitude: float | None, rng: random.Random) -> float:
    if latitude is None:
        return rng.uniform(-2.5, 2.5)
    hemisphere_scale = 1.0 if latitude >= 0 else -1.0
    return hemisphere_scale * rng.uniform(-2.0, 3.5)


def _sample_risk_core(label: str, rng: random.Random) -> float:
    if "wildfire" in label.lower() and not label.lower().startswith("no"):
        return _clip(rng.gauss(0.78, 0.1), 0.52, 0.98)
    return _clip(rng.gauss(0.22, 0.11), 0.02, 0.55)


def _simulate_weather(
    label: str,
    latitude: float | None,
    longitude: float | None,
    temporal_steps: int,
    rng: random.Random,
) -> SyntheticWeatherRecord:
    risk_core = _sample_risk_core(label, rng)
    season_offset = _seasonal_offset(latitude, rng)
    continentality = rng.uniform(-1.5, 1.5) + (abs(longitude or 0.0) % 3.0) * 0.15

    temperature_c = _clip(
        13.0 + (28.0 * risk_core) + season_offset + continentality + rng.gauss(0.0, 2.8),
        4.0,
        49.0,
    )
    humidity_pct = _clip(
        82.0 - (72.0 * risk_core) + rng.gauss(0.0, 7.5) - (season_offset * 0.8),
        5.0,
        95.0,
    )
    wind_speed_kph = _clip(
        4.0 + (30.0 * risk_core) + abs(rng.gauss(0.0, 5.0)),
        0.0,
        75.0,
    )
    drought_index = _clip(
        80.0
        + (690.0 * risk_core)
        + (temperature_c * 1.8)
        - (humidity_pct * 0.9)
        + rng.gauss(0.0, 28.0),
        0.0,
        800.0,
    )
    vegetation_dryness = _clip(
        0.18 + (0.78 * risk_core) + rng.gauss(0.0, 0.05),
        0.0,
        1.0,
    )
    days_since_rain = _clip(
        0.5 + (22.0 * risk_core) + rng.gauss(0.0, 2.2),
        0.0,
        45.0,
    )

    if "wildfire" in label.lower() and not label.lower().startswith("no"):
        temp_slope = rng.uniform(0.7, 2.1)
        humidity_slope = rng.uniform(2.0, 5.0)
        wind_slope = rng.uniform(0.8, 2.8)
        drought_slope = rng.uniform(8.0, 18.0)
        dryness_slope = rng.uniform(0.01, 0.035)
    else:
        temp_slope = rng.uniform(-0.3, 0.5)
        humidity_slope = rng.uniform(-1.2, 1.8)
        wind_slope = rng.uniform(-0.5, 0.8)
        drought_slope = rng.uniform(-3.0, 5.0)
        dryness_slope = rng.uniform(-0.01, 0.012)

    temporal_sequence: list[dict[str, float]] = []
    for step_index in range(temporal_steps):
        lag = temporal_steps - step_index - 1
        step_temperature = _clip(
            temperature_c - (temp_slope * lag) + rng.gauss(0.0, 0.8), 4.0, 49.0
        )
        step_humidity = _clip(
            humidity_pct + (humidity_slope * lag) + rng.gauss(0.0, 2.5),
            5.0,
            95.0,
        )
        step_wind = _clip(wind_speed_kph - (wind_slope * lag) + rng.gauss(0.0, 1.5), 0.0, 75.0)
        step_drought = _clip(
            drought_index - (drought_slope * lag) + rng.gauss(0.0, 6.0),
            0.0,
            800.0,
        )
        step_dryness = _clip(
            vegetation_dryness - (dryness_slope * lag) + rng.gauss(0.0, 0.015),
            0.0,
            1.0,
        )
        temporal_sequence.append(
            {
                "temperature_c": round(step_temperature, 3),
                "humidity_pct": round(step_humidity, 3),
                "wind_speed_kph": round(step_wind, 3),
                "drought_index": round(step_drought, 3),
                "vegetation_dryness": round(step_dryness, 4),
            }
        )

    return SyntheticWeatherRecord(
        temperature_c=round(temperature_c, 3),
        humidity_pct=round(humidity_pct, 3),
        wind_speed_kph=round(wind_speed_kph, 3),
        drought_index=round(drought_index, 3),
        vegetation_dryness=round(vegetation_dryness, 4),
        days_since_rain=round(days_since_rain, 3),
        latitude=round(latitude, 6) if latitude is not None else None,
        longitude=round(longitude, 6) if longitude is not None else None,
        risk_score=round(risk_core, 4),
        temporal_sequence=temporal_sequence,
    )


def _iter_labeled_images(root_dir: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                items.append((image_path, class_dir.name))
    return items


def _write_csv_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "label",
        "risk_score",
        "temperature_c",
        "humidity_pct",
        "wind_speed_kph",
        "drought_index",
        "vegetation_dryness",
        "days_since_rain",
        "latitude",
        "longitude",
        "temporal_sequence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def generate_manifest_for_split(
    split_dir: Path,
    output_dir: Path,
    split_name: str,
    temporal_steps: int,
    seed: int,
) -> tuple[Path, Path]:
    rng = random.Random(seed + int(math.fabs(hash(split_name)) % 100_000))
    rows: list[dict[str, object]] = []
    for image_path, label in _iter_labeled_images(split_dir):
        latitude, longitude = _parse_lat_lon(image_path)
        synthetic = _simulate_weather(
            label=label,
            latitude=latitude,
            longitude=longitude,
            temporal_steps=temporal_steps,
            rng=rng,
        )
        rows.append(
            {
                "image_path": str(image_path.as_posix()),
                "label": label,
                "risk_score": synthetic.risk_score,
                "temperature_c": synthetic.temperature_c,
                "humidity_pct": synthetic.humidity_pct,
                "wind_speed_kph": synthetic.wind_speed_kph,
                "drought_index": synthetic.drought_index,
                "vegetation_dryness": synthetic.vegetation_dryness,
                "days_since_rain": synthetic.days_since_rain,
                "latitude": synthetic.latitude,
                "longitude": synthetic.longitude,
                "temporal_sequence": json.dumps(synthetic.temporal_sequence),
            }
        )

    csv_path = output_dir / f"{split_name}_multimodal.csv"
    jsonl_path = output_dir / f"{split_name}_multimodal.jsonl"
    _write_csv_manifest(rows, csv_path)
    _write_jsonl_manifest(rows, jsonl_path)
    return csv_path, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multimodal wildfire manifests")
    parser.add_argument(
        "--data-root", default="data", help="Root directory containing train/valid/test"
    )
    parser.add_argument("--output-dir", default="data/manifests", help="Manifest output directory")
    parser.add_argument("--temporal-steps", type=int, default=4, help="Number of temporal steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for input_split, output_split in split_map.items():
        split_dir = data_root / input_split
        if not split_dir.exists():
            continue
        csv_path, jsonl_path = generate_manifest_for_split(
            split_dir=split_dir,
            output_dir=output_dir,
            split_name=output_split,
            temporal_steps=args.temporal_steps,
            seed=args.seed,
        )
        print(f"generated_csv={csv_path}")
        print(f"generated_jsonl={jsonl_path}")


if __name__ == "__main__":
    main()

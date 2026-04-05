import csv
import random
from pathlib import Path

from PIL import Image

from wildfire_mlops.cli.generate_multimodal_dataset import (
    _simulate_weather,
    generate_manifest_for_split,
)


def test_synthetic_weather_correlation():
    rng = random.Random(42)
    wildfire_samples = [
        _simulate_weather("wildfire", latitude=51.0, longitude=-114.0, temporal_steps=4, rng=rng)
        for _ in range(32)
    ]
    nowildfire_samples = [
        _simulate_weather("nowildfire", latitude=51.0, longitude=-114.0, temporal_steps=4, rng=rng)
        for _ in range(32)
    ]

    wildfire_temp = sum(item.temperature_c for item in wildfire_samples) / len(wildfire_samples)
    nofire_temp = sum(item.temperature_c for item in nowildfire_samples) / len(nowildfire_samples)
    wildfire_humidity = sum(item.humidity_pct for item in wildfire_samples) / len(wildfire_samples)
    nofire_humidity = sum(item.humidity_pct for item in nowildfire_samples) / len(
        nowildfire_samples
    )
    wildfire_wind = sum(item.wind_speed_kph for item in wildfire_samples) / len(wildfire_samples)
    nofire_wind = sum(item.wind_speed_kph for item in nowildfire_samples) / len(nowildfire_samples)

    assert wildfire_temp > nofire_temp
    assert wildfire_humidity < nofire_humidity
    assert wildfire_wind > nofire_wind


def test_generate_manifest_for_split(tmp_path: Path):
    split_dir = tmp_path / "train"
    (split_dir / "wildfire").mkdir(parents=True)
    (split_dir / "nowildfire").mkdir(parents=True)

    for class_name in ("wildfire", "nowildfire"):
        image_path = split_dir / class_name / "-113.9,51.0.jpg"
        Image.new("RGB", (32, 32), color=(128, 64, 32)).save(image_path)

    csv_path, jsonl_path = generate_manifest_for_split(
        split_dir=split_dir,
        output_dir=tmp_path / "manifests",
        split_name="train",
        temporal_steps=4,
        seed=42,
    )

    assert csv_path.exists()
    assert jsonl_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert "temperature_c" in rows[0]
    assert "temporal_sequence" in rows[0]

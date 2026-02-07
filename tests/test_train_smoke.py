from pathlib import Path

from wildfire_mlops.training import TrainConfig, train_model


def test_train_smoke(tmp_path: Path):
    cfg = TrainConfig(
        train_dir=Path("data/train"),
        val_dir=Path("data/valid"),
        test_dir=None,
        image_size=224,
        batch_size=8,
        num_workers=0,
        epochs=1,
        lr=0.0003,
        weight_decay=0.00001,
        device="cpu",
        output_dir=tmp_path,
        seed=42,
        max_train_samples=32,
        max_val_samples=16,
        mlflow_tracking_uri="./mlruns",
        mlflow_experiment="wildfire-test",
        mlflow_run_name="smoke",
        model_arch="custom_cnn",
        pretrained=False,
        register_model=False,
        model_registry_name="wildfire-classifier",
        max_stat_batches=2,
    )

    results = train_model(cfg)
    assert "best_val_accuracy" in results
    assert (tmp_path / "model_latest.pth").exists()

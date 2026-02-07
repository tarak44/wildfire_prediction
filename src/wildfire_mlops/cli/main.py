import argparse
from pathlib import Path

from PIL import Image

from wildfire_mlops.core import get_settings, setup_logging
from wildfire_mlops.inference import predict_image
from wildfire_mlops.modeling import load_checkpoint
from wildfire_mlops.pipelines import run_batch_inference
from wildfire_mlops.training import evaluate_dataset, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire prediction CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to image file")
    group.add_argument("--input-dir", help="Directory with images for batch inference")
    group.add_argument("--eval-dir", help="Root dir with class subfolders for evaluation")
    parser.add_argument("--output-csv", default="predictions.csv", help="Batch output CSV")
    parser.add_argument("--metrics-json", default="metrics.json", help="Eval metrics output JSON")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval samples")
    parser.add_argument("--model-path", default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--model-arch", default=None, help="Model architecture (custom_cnn or resnet18)"
    )
    parser.add_argument(
        "--pretrained", default=None, help="Use pretrained weights (true/false)"
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    model_path = args.model_path or settings.resolve_model_path(args.model_arch)
    model_arch = args.model_arch or settings.model_arch
    pretrained = (
        settings.pretrained
        if args.pretrained is None
        else str(args.pretrained).lower() == "true"
    )
    model, class_names = load_checkpoint(
        model_path,
        model_arch=model_arch,
        pretrained=pretrained,
    )

    if args.image:
        image_path = Path(args.image)
        image = Image.open(image_path).convert("RGB")

        pred = predict_image(
            image=image,
            model=model,
            class_names=class_names,
            device=settings.device,
            image_size=settings.image_size,
            reference_stats_path=settings.reference_stats_path,
        )

        print(f"class={pred.class_name} confidence={pred.confidence:.4f}")
        print(pred.probabilities)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_csv = Path(args.output_csv)
        count = run_batch_inference(
            input_dir=input_dir,
            output_csv=output_csv,
            model=model,
            class_names=class_names,
            device=settings.device,
            image_size=settings.image_size,
        )
        print(f"processed={count} output={output_csv}")
    else:
        eval_dir = Path(args.eval_dir)
        metrics = evaluate_dataset(
            root_dir=eval_dir,
            model=model,
            class_names=class_names,
            device=settings.device,
            image_size=settings.image_size,
            max_samples=args.max_eval_samples,
        )
        output_path = Path(args.metrics_json)
        save_metrics(metrics, output_path)
        print(f"metrics_saved={output_path}")


if __name__ == "__main__":
    main()

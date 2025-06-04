import argparse

from sea_clutter.generate_data import generate_segmentation_dataset
from sea_clutter.generate_data import generate_segmentation_dataset_with_sequences
from training.unet_training import train_model
from training.end_to_end import EndToEndTargetDetector, interactive_sample_explorer


def generate_dataset(args):
    if args.frames == 1:
        generate_segmentation_dataset(
            samples_per_class=args.samples,
            max_targets=args.max_targets,
            sea_state=args.sea_state,
            save_path=args.output,
        )
    else:
        generate_segmentation_dataset_with_sequences(
            samples_per_class=args.samples,
            max_targets=args.max_targets,
            sea_state=args.sea_state,
            n_frames=args.frames,
            save_path=args.output,
        )


def train_unet(args):
    train_model(
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=args.pretrained,
        model_save_path=args.output,
    )


def build_detector(args):
    if args.interactive:
        interactive_sample_explorer(args.dataset, args.model)
    else:
        EndToEndTargetDetector(unet_weights_path=args.model)
        print("End-to-end detector initialized. Use --interactive to explore samples.")


def main():
    parser = argparse.ArgumentParser(description="Reproduce Sea Clutter experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen.add_argument("--samples", type=int, default=500, help="Samples per class")
    gen.add_argument("--max-targets", type=int, default=5, help="Maximum targets")
    gen.add_argument("--sea-state", type=int, default=5, help="WMO sea state")
    gen.add_argument("--frames", type=int, default=1, help="Number of frames")
    gen.add_argument("--output", type=str, default="data/sea_clutter.pt", help="Output file")
    gen.set_defaults(func=generate_dataset)

    train = subparsers.add_parser("train-unet", help="Train U-Net model")
    train.add_argument("dataset", help="Path to dataset")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--pretrained", type=str, default=None)
    train.add_argument("--output", type=str, default="pretrained/unet.pt")
    train.set_defaults(func=train_unet)

    detect = subparsers.add_parser("detect", help="Build end-to-end detector")
    detect.add_argument("dataset", help="Dataset for evaluation")
    detect.add_argument("model", help="Trained U-Net weights")
    detect.add_argument("--interactive", action="store_true", help="Launch sample explorer")
    detect.set_defaults(func=build_detector)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()


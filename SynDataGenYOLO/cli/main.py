import argparse
import sys
from SynDataGenYOLO.cli import (
    data_generation_cli, dataset_mixer_cli, extract_cli,
    extract_labelme_from_folder_cli, extract_to_yolo_cli, show_images_with_bboxes_cli
)


def main():
    parser = argparse.ArgumentParser(
        prog="SynDataGenYOLO",
        description="Command line interface for synthetic data generation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Define available subcommands
    subcommands = {
        "generate": data_generation_cli.main,
        "mix": dataset_mixer_cli.main,
        "extract": extract_cli.main,
        "extract_labelme": extract_labelme_from_folder_cli.main,
        "extract_yolo": extract_to_yolo_cli.main,
        "show": show_images_with_bboxes_cli.main,
    }

    # Add subcommands without defining arguments here
    for cmd, func in subcommands.items():
        subparsers.add_parser(
            cmd, help=f"{cmd} command").set_defaults(func=func)

    args, remaining_args = parser.parse_known_args()
    print(f"Command: {args.command}, Remaining args: {remaining_args}")

    if hasattr(args, "func"):
        # Pass only the remaining args to subcommands
        args.func(remaining_args)
    else:
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()

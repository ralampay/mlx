import os
import random
import shutil
import typer
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from rich.table import Table
from rich.console import Console
from torchvision import transforms, datasets
from mlx.modules.datasets.one_shot_pair_dataset import OneShotPairDataset

console = Console()

def load_ic_one_shot_dataset(dataset_path, input_size=(105, 105), colored=True, n_pairs_per_class=100):
    """
    Load train and validation datasets for one-shot learning.

    Args:
        dataset_path (str): Base dataset directory with subfolders: 'train', 'val'
        input_size (tuple): (height, width) for resizing images
        colored (bool): Whether to load RGB images (True) or grayscale (False)
        n_pairs_per_class (int): Number of random pairs to generate per class

    Returns:
        (train_dataset, val_dataset)
    """
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Expected dataset structure:\n"
            f"{dataset_path}/train/<class_name>/img.png\n"
            f"{dataset_path}/val/<class_name>/img.png"
        )

    train_dataset = OneShotPairDataset(
        train_dir,
        input_size=input_size,
        colored=colored,
        n_pairs_per_class=n_pairs_per_class,
    )

    val_dataset = OneShotPairDataset(
        val_dir,
        input_size=input_size,
        colored=colored,
        n_pairs_per_class=n_pairs_per_class,
    )

    return train_dataset, val_dataset

def build_ic_one_shot(dataset_path: str):
    """
    Interactive dataset splitter for one-shot learning.
    Given a path like datasets/tagalog/, will create train/val/test splits per character.
    """

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        typer.secho(f"Dataset path not found: {dataset_path}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

    # Discover all label folders
    label_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    typer.secho(f"Found {len(label_dirs)} label(s) under {dataset_path.name}", fg=typer.colors.CYAN, bold=True)

    # Count images per label
    table = Table(title="Label Summary", show_lines=True)
    table.add_column("Label", style="cyan")
    table.add_column("Images", justify="right", style="magenta")

    label_counts = {}
    for label in label_dirs:
        count = len(list(label.glob("*.png")))
        label_counts[label.name] = count
        table.add_row(label.name, str(count))

    console.print(table)

    # Ask user how many to allocate for splits
    train_count = typer.prompt("How many images per label for TRAIN?", type=int)
    val_count = typer.prompt("How many images per label for VAL?", type=int)
    test_count = typer.prompt("How many images per label for TEST?", type=int)

    total_needed = train_count + val_count + test_count
    for label, count in label_counts.items():
        if count < total_needed:
            typer.secho(
                f"Label '{label}' has only {count} images — less than requested total {total_needed}.",
                fg=typer.colors.YELLOW,
            )

    # Ask for output directory
    output_path = Path(typer.prompt("Enter output path for split dataset"))
    if output_path.exists():
        typer.confirm(f"Output directory '{output_path}' already exists. Overwrite?", abort=True)
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create train/val/test folders
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(exist_ok=True)

    # Split each label
    typer.secho("Splitting dataset...", fg=typer.colors.BLUE, bold=True)
    for label in label_dirs:
        images = sorted(list(label.glob("*.png")))
        random.shuffle(images)

        splits = {
            "train": images[:train_count],
            "val": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:train_count + val_count + test_count],
        }

        for split, split_images in splits.items():
            out_dir = output_path / split / label.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, out_dir / img_path.name)

    typer.secho(f"Dataset created successfully at {output_path}", fg=typer.colors.GREEN, bold=True)

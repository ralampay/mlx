"""Load prepared detection manifests independently of their source dataset."""
import json
from pathlib import Path
from mlx.core.exceptions import MLXUserError

def load_prepared_adapter_dataset(source: Path, *, expected_classes=None) -> dict:
    source = Path(source).expanduser().resolve()
    manifest_path = source / "manifest.json"
    yaml_path = source / "data.yaml"
    if not manifest_path.is_file() or not yaml_path.is_file():
        raise MLXUserError(
            f"Prepared adapter dataset requires manifest.json and data.yaml at {source}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MLXUserError(f"Cannot read prepared dataset manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise MLXUserError("Prepared dataset manifest must be an object")
    if expected_classes is not None and manifest.get("classes") != list(expected_classes):
        raise MLXUserError("Prepared dataset class order does not match the foundation taxonomy")
    classes = manifest.get("classes")
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(name, str) or not name.strip() for name in classes)
            or len(set(classes)) != len(classes)):
        raise MLXUserError("Prepared dataset requires unique, nonempty class names")
    if expected_classes is None:
        import yaml
        try:
            configuration = yaml.safe_load(yaml_path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise MLXUserError(f"Cannot read dataset YAML {yaml_path}: {exc}") from exc
        if not isinstance(configuration, dict):
            raise MLXUserError("Dataset YAML must be a mapping with class names")
        names = configuration.get("names")
        if isinstance(names, dict):
            names = [names.get(index) for index in range(len(names))]
        if names != classes:
            raise MLXUserError("Dataset YAML class order differs from manifest")
    for split in ("train", "val", "test"):
        if not (source / "images" / split).is_dir() or not (source / "labels" / split).is_dir():
            raise MLXUserError(f"Prepared dataset is missing {split} images or labels")
    return manifest


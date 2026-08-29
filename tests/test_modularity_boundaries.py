from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODES = ROOT / "mlx" / "modes"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_workflow_modules_do_not_depend_on_terminal_presentation() -> None:
    presentation_files = {
        path.resolve()
        for path in MODES.rglob("presentation.py")
    }
    composition_files = {
        path.resolve()
        for pattern in ("runner.py", "entrypoint.py")
        for path in MODES.rglob(pattern)
    }

    violations: list[str] = []
    for path in MODES.rglob("*.py"):
        if path.resolve() in presentation_files | composition_files:
            continue
        for module in _imports(path):
            if module == "rich" or module.startswith("rich.") or module == "mlx.core.ui":
                violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert violations == []


def test_segmentation_encoder_uses_an_explicit_backbone_adapter() -> None:
    encoder_imports = _imports(MODES / "segmentation" / "models" / "backbones.py")
    adapter_imports = _imports(
        MODES / "segmentation" / "models" / "backbone_factory.py"
    )

    assert not any(
        module.startswith("mlx.modes.image_classification")
        for module in encoder_imports
    )
    assert "mlx.modes.image_classification.models" in adapter_imports


def test_aws_integration_has_separate_client_image_and_status_boundaries() -> None:
    for mode in ("object_detection", "image_classification"):
        aws_package = MODES / mode / "aws"
        assert (aws_package / "clients.py").is_file()
        assert (aws_package / "image.py").is_file()
        assert (aws_package / "status.py").is_file()


def test_classification_aws_boundary_does_not_depend_on_detection_mode() -> None:
    aws_package = MODES / "image_classification" / "aws"
    violations = []
    for path in aws_package.glob("*.py"):
        for module in _imports(path):
            if module.startswith("mlx.modes.object_detection"):
                violations.append(f"{path.name} imports {module}")

    assert violations == []

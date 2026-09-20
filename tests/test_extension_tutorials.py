"""Execute tutorial Python examples against the public APIs they document."""
from pathlib import Path
import re
import pytest

TUTORIALS = Path(__file__).resolve().parents[1] / "docs/tutorials"


@pytest.mark.parametrize("path", sorted(TUTORIALS.glob("adding-*.md")), ids=lambda p: p.stem)
def test_tutorial_python_blocks(path):
    namespace = {}
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.DOTALL)
    assert blocks, f"{path.name} must contain a working example."
    for index, source in enumerate(blocks):
        exec(compile(source, f"{path.name}:block{index}", "exec"), namespace)

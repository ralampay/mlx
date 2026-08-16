from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlx.modes.object_detection.ultralytics.adapters import (
    _result_to_detection_result,
)


class FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBoxes:
    xyxy = FakeTensor([[1.25, 2.5, 30.75, 40.125]])
    conf = FakeTensor([0.8])
    cls = FakeTensor([0])

    def __len__(self) -> int:
        return 1


def test_ultralytics_adapter_preserves_fractional_boxes() -> None:
    result = _result_to_detection_result(
        SimpleNamespace(names={0: "person"}, boxes=FakeBoxes())
    )

    assert result.detections[0].xyxy == pytest.approx((1.25, 2.5, 30.75, 40.125))

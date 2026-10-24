"""Explicitly opt-in public catalogue smoke test; never downloads an archive."""
import os
import pytest


@pytest.mark.skipif(os.environ.get("MLX_RUN_ASF_LIVE") != "1", reason="Set MLX_RUN_ASF_LIVE=1 for public ASF network access")
def test_public_asf_catalogue_contract():
    pytest.importorskip("asf_search")
    from mlx.modes.sar.asf.provider import AsfProvider
    from mlx.modes.sar.requests import BoundingBox, SearchRequest, NetworkOptions
    scenes = AsfProvider(network=NetworkOptions(timeout=30, max_retries=1)).search(
        SearchRequest(BoundingBox(0, 0, 1, 1), "2024-01-01", "2024-02-01", max_results=1))
    assert len(scenes) <= 1
    for scene in scenes:
        assert scene.processing_level == "GRD_HD"
        assert scene.filename.endswith(".zip")
        assert scene.metadata["properties"]["fileID"] == scene.scene_id

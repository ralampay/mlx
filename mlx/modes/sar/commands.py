"""SAR use cases; external providers and cache storage are injected."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from mlx.core.artifacts import write_json_atomic
from mlx.core.commands import NullWorkflowReporter, WorkflowEvent
from mlx.core.exceptions import MLXUserError
from .models import SearchResult, DownloadResult
from .selection import rank_scenes


class SearchSarScenes:
    def __init__(self, request, provider, output, reporter=None):
        self.request, self.provider, self.output = request, provider, output
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self):
        self.reporter.emit(WorkflowEvent("info", "Searching public SAR catalogue"))
        candidates = self.provider.search(self.request)
        result = SearchResult(rank_scenes(candidates, self.request), len(candidates), self.request.max_results,
                              query=asdict(self.request))
        write_json_atomic(self.output / "metadata" / "scenes.json", result)
        return result


class DownloadSarScenes:
    def __init__(self, request, provider, cache, reporter=None):
        self.request, self.provider, self.cache = request, provider, cache
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self):
        try:
            return self._execute()
        except OSError as exc:
            raise MLXUserError("Unable to store SAR archives. Check cache/output permissions and free disk space.") from exc

    def _execute(self):
        request = self.request
        found = SearchSarScenes(request.search, self.provider, request.output, self.reporter).execute()
        selected = found.scenes if request.all_scenes else found.scenes[:1]
        if not selected:
            raise MLXUserError("No matching Sentinel-1 GRD_HD scenes. Adjust dates, AOI or polarization.")
        if len({s.filename for s in selected}) != len(selected):
            raise MLXUserError("Catalogue returned conflicting archive filenames; select one --scene-id at a time.")
        if request.dry_run:
            plan = tuple({"scene_id": s.scene_id, "url": s.url, "expected_size_bytes": s.size_bytes,
                          "size_is_exact": s.size_is_exact,
                          "path": str(request.output / "raw" / s.filename)} for s in selected)
            return DownloadResult(selected, plan, True)
        if request.processes < 1:
            raise MLXUserError("Download processes must be positive.")
        self.reporter.emit(WorkflowEvent("info", f"Downloading {len(selected)} selected scene(s)"))
        def download(scene):
            return self.cache.obtain(scene, self.provider, request.output)
        records = []
        index_path = request.output / "metadata" / "downloads.json"
        try:
            previous = json.loads(index_path.read_text()) if index_path.exists() else []
            inventory = {record["scene_id"]: record for record in previous}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise MLXUserError("Existing downloads.json is invalid; move it aside before downloading.") from exc
        with ThreadPoolExecutor(max_workers=min(request.processes, len(selected))) as executor:
            for record in executor.map(download, selected):
                records.append(record)
                inventory[record["scene_id"]] = record
                self.reporter.emit(WorkflowEvent("info", f"Verified {record['scene_id']}", len(records), len(selected)))
                # Record each completed acquisition even if a later download fails.
                write_json_atomic(index_path, [inventory[key] for key in sorted(inventory)])
        return DownloadResult(selected, tuple(records), False)

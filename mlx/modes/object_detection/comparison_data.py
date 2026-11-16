"""Non-destructive, content-audited YOLO comparison partitions."""
from collections import Counter
import hashlib
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import yaml

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError


class PrepareComparisonDataset:
    def __init__(self, source, output):
        self.source, self.output = Path(source).resolve(), Path(output).resolve()

    def execute(self):
        if self.output.exists():
            raise MLXUserError(f"Dataset output already exists: {self.output}")
        spec = yaml.safe_load((self.source / 'data.yaml').read_text())
        names = spec['names']
        if isinstance(names, dict):
            names = [names[i] for i in range(len(names))]
        root = Path(spec.get('path', self.source))
        if not root.is_absolute():
            root = self.source / root
        seen, kept, excluded, thumbnails = {}, [], [], {}
        for split in ('test', 'val', 'train'):
            directory = root / spec[split]
            if not directory.is_dir():
                raise MLXUserError(f"Expected a YOLO image directory: {directory}")
            for image in sorted(directory.rglob('*')):
                if image.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                relative = image.relative_to(directory)
                label = root / 'labels' / split / relative.with_suffix('.txt')
                if not label.is_file():
                    raise MLXUserError(f"Missing annotation: {label}")
                objects = []
                for line in label.read_text().splitlines():
                    values = [float(v) for v in line.split()]
                    if len(values) != 5 or not np.isfinite(values).all():
                        raise MLXUserError(f"Invalid bounding box in {label}")
                    c, x, y, w, h = values
                    if c != int(c) or not 0 <= c < len(names) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise MLXUserError(f"Out-of-range bounding box in {label}")
                    objects.append(int(c))
                with Image.open(image) as im:
                    rgb = im.convert('RGB')
                    decoded = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    small = np.asarray(rgb.convert('L').resize((9, 8)), dtype=np.int16)
                    dhash = sum(int(b) << i for i, b in enumerate((small[:, 1:] > small[:, :-1]).flat))
                    thumbnails[str(image)] = np.asarray(rgb.resize((64,64)), dtype=np.float32)/255
                record = dict(split=split, relative=str(relative), source=str(image), label=str(label),
                              sha256=sha256_file(image), label_sha256=sha256_file(label), decoded=decoded,
                              dhash=dhash, objects=objects)
                if decoded in seen:
                    excluded.append(dict(record, duplicate_of=seen[decoded]['source']))
                else:
                    seen[decoded] = record
                    kept.append(record)
        suspicious = []
        for i, a in enumerate(kept):
            for b in kept[:i]:
                if (a['dhash'] ^ b['dhash']).bit_count() <= 2 and float(np.mean((thumbnails[a['source']]-thumbnails[b['source']])**2)) < .001:
                    suspicious.append([a['source'], b['source']])
        # Conservatively treat near-identical candidates as one image group.
        # The earlier priority order preserves test, then validation, then train.
        parents = {r['source']:r['source'] for r in kept}
        def representative(value):
            while parents[value] != value:
                value = parents[value]
            return value
        for a,b in suspicious:
            parents[representative(a)] = representative(b)
        groups, filtered = set(), []
        for row in kept:
            group = representative(row['source'])
            if group in groups:
                excluded.append(dict(row, reason='conservative-near-duplicate-group', group=group))
            else:
                groups.add(group)
                filtered.append(row)
        kept = filtered
        counts = {}
        for split in ('train', 'val', 'test'):
            rows = [r for r in kept if r['split'] == split]
            objects = Counter(c for r in rows for c in r['objects'])
            if not rows or set(objects) != set(range(len(names))):
                raise MLXUserError(f"Cleaned {split} does not represent all classes")
            counts[split] = dict(images=len(rows), objects_per_class={names[c]: objects[c] for c in objects})
        self.output.mkdir(parents=True)
        for row in kept:
            for kind, source in (('images', row['source']), ('labels', row['label'])):
                relative = Path(row['relative'])
                destination = self.output / kind / row['split'] / (relative if kind == 'images' else relative.with_suffix('.txt'))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        spec = dict(path=str(self.output), names=names, nc=len(names), **{s: f'images/{s}' for s in counts})
        (self.output / 'data.yaml').write_text(yaml.safe_dump(spec))
        manifest = dict(source=str(self.source), names=names, counts=counts, kept=kept,
                        excluded=excluded, near_duplicate_pairs=suspicious,
                        near_duplicate_policy='Connected groups: dHash Hamming <=2 and RGB 64x64 MSE <0.001; keep one by test/val/train priority. Conservative exclusion may remove distinct similar textures.',
                        group_metadata='Not supplied: source-scene independence not established',
                        status='audited')
        write_json_atomic(self.output / 'audit.json', manifest)
        return manifest

"""Sentinel-1 sigma-nought LUT interpolation, without terrain/noise correction."""
from mlx.core.exceptions import MLXUserError


class SigmaCalibration:
    def __init__(self, xml):
        import numpy as np
        from defusedxml.ElementTree import fromstring
        try:
            root = fromstring(xml)
            vectors = []
            for vector in root.findall(".//{*}calibrationVector"):
                line = int(vector.findtext("{*}line"))
                pixels = np.fromstring(vector.findtext("{*}pixel"), sep=" ")
                values = np.fromstring(vector.findtext("{*}sigmaNought"), sep=" ")
                if (len(pixels) < 2 or len(pixels) != len(values) or not np.all(np.diff(pixels) > 0)
                        or not np.all(np.isfinite(pixels))
                        or not np.all(np.isfinite(values)) or np.any(values <= 0)):
                    raise ValueError("Invalid sigma LUT")
                vectors.append((line, pixels, values))
            self.vectors = sorted(vectors, key=lambda row: row[0])
            self.lines = np.array([v[0] for v in self.vectors])
            if len(self.lines) < 2 or not np.all(np.diff(self.lines) > 0):
                raise ValueError("Invalid calibration lines")
        except Exception as exc:
            raise MLXUserError("Invalid SAFE sigmaNought calibration metadata; calibrated output cannot be produced.") from exc

    def apply(self, dn, row, col, *, decibels=False):
        import numpy as np
        rows = np.arange(row, row + dn.shape[0])
        columns = np.arange(col, col + dn.shape[1])
        if rows[0] < self.lines[0] or rows[-1] > self.lines[-1]:
            raise MLXUserError("Calibration LUT does not cover the requested image lines.")
        output = np.empty(dn.shape, dtype=np.float32)
        for index, line in enumerate(rows):
            upper = min(max(int(np.searchsorted(self.lines, line, side="right")), 1), len(self.lines) - 1)
            lower = upper - 1
            interpolated = []
            for vector in (self.vectors[lower], self.vectors[upper]):
                if columns[0] < vector[1][0] or columns[-1] > vector[1][-1]:
                    raise MLXUserError("Calibration LUT does not cover the requested range pixels.")
                interpolated.append(np.interp(columns, vector[1], vector[2]))
            weight = (line - self.lines[lower]) / (self.lines[upper] - self.lines[lower])
            lut = interpolated[0] * (1 - weight) + interpolated[1] * weight
            linear = np.square(dn[index].astype(np.float64) / lut)
            valid = dn[index] > 0
            output[index] = np.nan
            output[index, valid] = 10 * np.log10(linear[valid]) if decibels else linear[valid]
        return output

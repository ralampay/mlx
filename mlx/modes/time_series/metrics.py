"""Temporal fidelity, diversity and next-step utility diagnostics."""

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


def acf(values, max_lag=None):
    centered = values - values.mean(axis=1, keepdims=True)
    variance = np.mean(centered**2, axis=(0, 1))
    max_lag = min(20, values.shape[1] - 1) if max_lag is None else max_lag
    result = []
    for lag in range(1, max_lag + 1):
        covariance = np.mean(centered[:, :-lag] * centered[:, lag:], axis=(0, 1))
        result.append(
            np.divide(
                covariance,
                variance,
                out=np.full_like(variance, np.nan),
                where=variance > 1e-12,
            )
        )
    return np.asarray(result)


def spectrum(values):
    centered = values - values.mean(axis=1, keepdims=True)
    power = (np.abs(np.fft.rfft(centered, axis=1)) ** 2).mean(axis=0)
    totals = power.sum(axis=0)
    return np.divide(
        power, totals, out=np.full_like(power, np.nan), where=totals > 1e-12
    )


def finite_mean(values):
    values = np.asarray(values)
    return (
        float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None
    )


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-12 else None


def distribution_metrics(real, synthetic, features):
    real_acf, synth_acf = acf(real), acf(synthetic)
    real_psd, synth_psd = spectrum(real), spectrum(synthetic)
    per_feature = {}
    for index, name in enumerate(features):
        r, s = real[:, :, index], synthetic[:, :, index]
        per_feature[name] = {
            "wasserstein": float(wasserstein_distance(r.ravel(), s.ravel())),
            "ks_statistic": float(ks_2samp(r.ravel(), s.ravel()).statistic),
            "std_ratio": ratio(s.std(), r.std()),
            "within_window_std_ratio": ratio(
                s.std(axis=1).mean(), r.std(axis=1).mean()
            ),
            "between_window_mean_std_ratio": ratio(
                s.mean(axis=1).std(), r.mean(axis=1).std()
            ),
            "first_difference_std_ratio": ratio(
                np.diff(s, axis=1).std(), np.diff(r, axis=1).std()
            ),
            "acf_mae": finite_mean(np.abs(real_acf[:, index] - synth_acf[:, index])),
            "spectrum_l1": float(np.abs(real_psd[:, index] - synth_psd[:, index]).sum())
            if np.isfinite(real_psd[:, index]).all()
            and np.isfinite(synth_psd[:, index]).all()
            else None,
        }
    correlations = []
    for left in range(len(features)):
        for right in range(left + 1, len(features)):
            r, s = real.reshape(-1, len(features)), synthetic.reshape(-1, len(features))
            if (
                min(
                    r[:, left].std(),
                    r[:, right].std(),
                    s[:, left].std(),
                    s[:, right].std(),
                )
                > 1e-12
            ):
                correlations.append(
                    abs(
                        np.corrcoef(r[:, left], r[:, right])[0, 1]
                        - np.corrcoef(s[:, left], s[:, right])[0, 1]
                    )
                )
    aggregate = {
        key: finite_mean(
            [v[key] if v[key] is not None else np.nan for v in per_feature.values()]
        )
        for key in next(iter(per_feature.values()))
    }
    aggregate["cross_feature_correlation_mae"] = finite_mean(correlations)
    return {"per_feature": per_feature, "aggregate": aggregate}


def descriptive(values, features):
    return {
        name: {
            "mean": float(values[:, :, i].mean()),
            "std": float(values[:, :, i].std()),
            "quantiles_05_50_95": np.quantile(
                values[:, :, i], [0.05, 0.5, 0.95]
            ).tolist(),
        }
        for i, name in enumerate(features)
    }


def nearest_distances(query, reference, exclude_self=False):
    q, r = query.reshape(len(query), -1), reference.reshape(len(reference), -1)
    if exclude_self and len(r) < 2:
        return None
    neighbors = NearestNeighbors(n_neighbors=2 if exclude_self else 1).fit(r)
    distances = neighbors.kneighbors(q)[0][:, -1] / np.sqrt(q.shape[1])
    return {
        "mean": float(distances.mean()),
        "quantiles_05_50_95": np.quantile(distances, [0.05, 0.5, 0.95]).tolist(),
    }


def predictive_utility(train, real, synthetic, rng):
    def evaluate(source):
        # Identical fixed one-lag ridge predictor for both training sources.
        model = Ridge(alpha=1.0).fit(
            source[:, :-1].reshape(-1, source.shape[2]),
            source[:, 1:].reshape(-1, source.shape[2]),
        )
        errors = model.predict(real[:, :-1].reshape(-1, real.shape[2])) - real[
            :, 1:
        ].reshape(-1, real.shape[2])
        return {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
        }

    # Match training-window counts so dataset size is not the source of a utility gap.
    count = min(len(train), len(synthetic))
    real_score = evaluate(train[rng.choice(len(train), count, replace=False)])
    synth_score = evaluate(synthetic[rng.choice(len(synthetic), count, replace=False)])
    return {
        "predictor": "ridge(alpha=1), one-step history; standardized units",
        "training_windows_each": count,
        "train_real_test_real": real_score,
        "train_synthetic_test_real": synth_score,
        "synthetic_minus_real": {
            key: synth_score[key] - real_score[key] for key in real_score
        },
    }


def bootstrap_indices(groups, rng):
    unique = np.unique(groups)
    if len(unique) > 1:
        return np.concatenate(
            [
                np.flatnonzero(groups == group)
                for group in rng.choice(unique, len(unique), replace=True)
            ]
        )
    count = len(groups)
    block = max(1, int(np.sqrt(count)))
    starts = rng.integers(
        0, max(1, count - block + 1), size=(count + block - 1) // block
    )
    return np.concatenate([np.arange(start, start + block) for start in starts])[:count]


class TemporalQualityEvaluator:
    def evaluate(
        self,
        training,
        validation,
        real,
        synthetic,
        groups,
        metadata,
        request,
        *,
        training_reference=None,
    ):
        rng = np.random.default_rng(request.random_seed)

        # Real evaluation uses a contiguous span, so block bootstrap never joins gaps.
        def subset(values):
            indices = np.sort(
                rng.choice(
                    len(values),
                    min(len(values), request.max_metric_windows),
                    replace=False,
                )
            )
            return values[indices], indices

        original_counts = {
            "training": len(training),
            "validation": len(validation),
            "real": len(real),
            "synthetic": len(synthetic),
        }
        training_reference = (
            training if training_reference is None else training_reference
        )
        original_counts["training_reference"] = len(training_reference)
        training_reference, _ = subset(training_reference)
        training, _ = subset(training)
        validation, _ = subset(validation)
        count = min(len(real), request.max_metric_windows)
        start = int(rng.integers(0, len(real) - count + 1))
        selected = np.arange(start, start + count)
        real = real[selected]
        groups = groups[selected]
        synthetic, _ = subset(synthetic)
        features = metadata["feature_columns"]
        main = distribution_metrics(real, synthetic, features)
        samples = {key: [] for key in main["aggregate"]}
        for _ in range(request.bootstrap_samples):
            indices = bootstrap_indices(groups, rng)
            generated = rng.integers(0, len(synthetic), len(synthetic))
            scores = distribution_metrics(
                real[indices], synthetic[generated], features
            )["aggregate"]
            for key, value in scores.items():
                if value is not None:
                    samples[key].append(value)
        intervals = {
            key: np.quantile(values, [0.025, 0.975]).tolist() if values else None
            for key, values in samples.items()
        }
        scale, mean = np.asarray(metadata["scale"]), np.asarray(metadata["mean"])
        return {
            "fidelity": main,
            "real_vs_real_reference": distribution_metrics(real, validation, features),
            "bootstrap_95_percent_intervals": intervals,
            "descriptive_original_units": {
                "real": descriptive(real * scale + mean, features),
                "synthetic": descriptive(synthetic * scale + mean, features),
            },
            "copying": {
                "duplicate_fraction": float(
                    1
                    - len(np.unique(synthetic.reshape(len(synthetic), -1), axis=0))
                    / len(synthetic)
                ),
                "synthetic_to_training": nearest_distances(
                    synthetic, training_reference
                ),
                "heldout_real_to_training": nearest_distances(real, training_reference),
                "synthetic_to_synthetic": nearest_distances(synthetic, synthetic, True),
            },
            "utility": predictive_utility(training, real, synthetic, rng),
            "protocol": {
                "original_window_counts": original_counts,
                "evaluated_window_counts": {
                    "training": len(training),
                    "validation": len(validation),
                    "real": len(real),
                    "synthetic": len(synthetic),
                    "training_reference": len(training_reference),
                },
                "seed": request.random_seed,
                "bootstrap_samples": request.bootstrap_samples,
                "real_window_selection": "seeded contiguous span in series/time order",
                "real_selected_span": [start, start + count],
                "max_metric_windows": request.max_metric_windows,
                "bootstrap_unit": "series clusters"
                if len(np.unique(groups)) > 1
                else "moving blocks of evaluation windows",
                "distance_units": "training-standardized; nearest distances are window RMSE",
                "undefined_metrics": "null means insufficient variation or samples",
                "limitations": "No universal quality threshold or privacy guarantee. Real-vs-real uses validation vs test and can reflect drift. Subsampled windows limit copying detection. Intervals describe sampling variation, not training-seed variation.",
            },
        }

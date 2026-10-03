"""Fixed v2 pilot and confirmation protocols, with no runtime environment reads."""
from mlx.modes.text_embedding.retrieval_datasets import SUITE_DATASETS


def _variant(name, model="simple", loss="mse", **options):
    return {"name": name, "model": model, "loss": loss, "dimensions": [384, 512], **options}


def _training():
    return {"hidden_dim": 512, "epochs": 50, "batch_size": 64, "lr": 0.001, "val_ratio": 0.2, "device": "cpu"}


def _ordered():
    return _variant("ordered", "ordered-simple", dimensions=[512], evaluation_dimensions=[384, 512],
                    model_config={"prefix_dimensions": [128, 256, 384, 512]})


def pilot_config():
    variants = [_variant("mse"), _variant("spectral-mse", "simple-spectral")]
    for rho in (0.01, 0.1, 1.0):
        suffix = str(rho).replace(".", "p")
        variants += [_variant(f"least-volume-{suffix}", "simple-spectral", "mse-least-volume", loss_config={"rho": rho, "eta": 1e-3}),
                     _variant(f"covariance-{suffix}", loss="mse-covariance", loss_config={"rho": rho})]
    variants.append(_ordered())
    return {"schema_version": 2, "phase": "pilot", "datasets": ["nfcorpus", "nano-quora", "nano-fiqa2018"],
            "seeds": [40, 41], "training": _training(), "variants": variants, "secondary": []}


def confirmation_config(selected):
    return {"schema_version": 2, "phase": "confirmation", "datasets": list(SUITE_DATASETS),
            "seeds": [42, 43, 44, 45, 46], "training": _training(), "variants": [
                _variant("mse"), _variant("mse-similarity", loss="mse-similarity", loss_config={"similarity_weight": 1.0}),
                _variant("spectral-mse", "simple-spectral"),
                _variant("least-volume", "simple-spectral", "mse-least-volume", loss_config={"rho": selected["mse-least-volume"], "eta": 1e-3}),
                _variant("covariance", loss="mse-covariance", loss_config={"rho": selected["mse-covariance"]}),
                _ordered(), {"name": "pca", "kind": "pca", "dimensions": [512], "evaluation_dimensions": [384, 512]},
                {"name": "truncate", "kind": "truncate", "dimensions": [384, 512]},
            ], "secondary": [["least-volume", "spectral-mse"], ["covariance", "mse"],
                              ["ordered", "mse"], ["mse-similarity", "mse"]]}


def orthogonal_config():
    """Frozen exploratory comparison; retrieval scores must not retune this recipe."""
    losses = ["orthogonal-mse", "orthogonal-cosine"]
    return {"schema_version": 2, "phase": "confirmation", "datasets": list(SUITE_DATASETS),
            "seeds": [42, 43, 44, 45, 46], "training": _training(),
            "variants": [
                _variant(losses[0], "orthogonal-tied"),
                _variant(losses[1], "orthogonal-tied", "mse-cosine", loss_config={"cosine_weight": 1.0}),
                {"name": "pca", "kind": "pca", "dimensions": [512], "evaluation_dimensions": [384, 512]},
                {"name": "svd", "kind": "svd", "dimensions": [512], "evaluation_dimensions": [384, 512]},
                {"name": "truncate", "kind": "truncate", "dimensions": [384, 512]},
            ], "secondary": [[losses[1], losses[0]]] + [[loss, control] for loss in losses for control in ("pca", "svd")],
            "selection": {"interpretation": "exploratory: all ten datasets have already informed model selection",
                          "primary_metric": "ndcg@10", "noninferiority_margin": 0.01,
                          "loss_weight": "fixed before retrieval; cosine_weight=1, divided by input dimension"}}

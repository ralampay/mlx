# Extending MLX

Start with [the extension inventory](../extensions.md) and [the architecture](../../ARCHITECTURE.md).
Use explicit, local registries in Python. Registering an alias in one Python process does not
install it into future CLI processes. For a permanent built-in alias, change the owning catalog;
where supported, CLI import references select installed external code without global registration.

These tutorials teach framework plumbing, not competitive algorithms. All Python blocks are
executed by `tests/test_extension_tutorials.py`. Run from the repository root after installing
MLX's normal dependencies. Shell training examples additionally require the documented datasets.

- [Autoencoder](adding-an-autoencoder.md)
- [Tracker](adding-a-tracker.md)
- [Loss](adding-a-loss.md)
- [Metric](adding-a-metric.md)
- [Classifier](adding-a-classification-model.md)
- [Segmentation and saliency model](adding-a-segmentation-model.md)
- [Detection adapter](adding-an-object-detection-adapter.md)

A normal extension changes its implementation, its local registration (or import reference),
tests, and documentation—not CLI routing or unrelated workflows.

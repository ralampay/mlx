"""Scoped handling of optional provider import diagnostics."""

from contextlib import contextmanager
import logging


class _OptionalAttentionImportFilter(logging.Filter):
    def filter(self, record):
        return record.getMessage() != "xFormers not available"


@contextmanager
def quiet_unused_attention_imports():
    """Quiet DINOv2 availability notices while loading detection models."""
    logger = logging.getLogger("dinov2")
    diagnostic_filter = _OptionalAttentionImportFilter()
    logger.addFilter(diagnostic_filter)
    try:
        yield
    finally:
        logger.removeFilter(diagnostic_filter)

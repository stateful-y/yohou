"""Scoring context for carrying metadata through the scorer pipeline.

``ScoringContext`` lives in ``yohou.utils._context`` (it is validation metadata,
a ``utils`` concern). It is re-exported here for backward compatibility so
``from yohou.metrics._context import ScoringContext`` keeps working.
"""

from __future__ import annotations

from yohou.utils._context import ScoringContext

__all__ = ["ScoringContext"]

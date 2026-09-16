"""The default ``validation=None`` search reproduces results recorded before ``validation="cv"`` existed."""

import json
from pathlib import Path

from .search_snapshot_case import build_search, summarize

SNAPSHOT = Path(__file__).parent / "data" / "search_validation_none_snapshot.json"


def test_default_search_matches_snapshot():
    search, y = build_search()
    search.fit(y, forecasting_horizon=3)
    observed = json.loads(json.dumps(summarize(search), sort_keys=True))
    expected = json.loads(SNAPSHOT.read_text())
    assert observed["cv_results_keys"] == expected["cv_results_keys"]
    assert observed["cv_results"] == expected["cv_results"]
    assert observed["best_params"] == expected["best_params"]
    assert observed["best_score"] == expected["best_score"]
    assert observed["best_index"] == expected["best_index"]
    assert observed["predictions"] == expected["predictions"]

from typing import List, Tuple

from tools.testing.target_determination.heuristics.correlated_with_historical_failures import (
    CorrelatedWithHistoricalFailures,
)
from tools.testing.target_determination.heuristics.edited_by_pr import EditedByPR
from tools.testing.target_determination.heuristics.historical_class_failure_correlation import (
    HistoricalClassFailurCorrelation,
)
from tools.testing.target_determination.heuristics.historical_edited_files import (
    HistorialEditedFiles,
)

from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics as AggregatedHeuristics,
    HeuristicInterface as HeuristicInterface,
    TestPrioritizations as TestPrioritizations,
)
from tools.testing.target_determination.heuristics.mentioned_in_pr import MentionedInPR

from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    PreviouslyFailedInPR,
)
from tools.testing.target_determination.heuristics.profiling import Profiling

# All currently running heuristics.
# To add a heurstic in trial mode, specify the keywork argument `trial_mode=True`.
HEURISTICS: List[Tuple[HeuristicInterface, float]] = [
    (PreviouslyFailedInPR(), 1.0),
    (EditedByPR(), 1.0),
    (MentionedInPR(), 1.0),
    (HistoricalClassFailurCorrelation(trial_mode=True), 0.25),
    (CorrelatedWithHistoricalFailures(), 0.25),
    (HistorialEditedFiles(), 0.25),
    (Profiling(), 0.25),
]

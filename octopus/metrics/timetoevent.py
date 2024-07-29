"""Time to event metrics."""

from sksurv.metrics import concordance_index_censored

# Constants for metric names
CI = "CI"

timetoevent_metrics = {
    CI: concordance_index_censored,
}

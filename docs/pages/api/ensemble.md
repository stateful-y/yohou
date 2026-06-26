---
template: api-submodule.html
---

# yohou.ensemble

Ensemble forecasters for combining predictions from multiple base forecasters.

**User guide**: See the [Ensemble Forecasting](../explanation/ensemble-forecasting.md) section for design rationale and usage patterns.

### Forecasters

| Name | Description |
| --- | --- |
| [`VotingPointForecaster`](generated/yohou.ensemble.voting_point.VotingPointForecaster.md) | Combines point predictions via mean or median (optionally weighted). |
| [`VotingIntervalForecaster`](generated/yohou.ensemble.voting_interval.VotingIntervalForecaster.md) | Combines interval predictions via mean, median, or envelope (mean optionally weighted). |
| [`VotingClassProbaForecaster`](generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster.md) | Combines class-probability forecasters via soft or hard voting. |

# User Guide

## Estimator

The central piece of transformer, regressor, and classifier is `sklearn.base.BaseEstimator`. All estimators in scikit-learn are derived from this class. In more details, this base class enables to set and get parameters of the estimator. It can be imported as:

```python
from sklearn.base import BaseEstimator
```

Once imported, you can create a class which inherits from this base class.

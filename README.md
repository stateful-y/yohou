yohou - A time series forecasting package based on scikit-learn and polars
==========================================================================

[![Deploy PyPI](https://github.com/gtauzin/yohou/actions/workflows/deploy-pypi.yml/badge.svg?event=push)](https://github.com/gtauzin/yohou/actions/workflows/deploy-pypi.yml)
![tests](https://github.com/gtauzin/yohou/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/gtauzin/yohoue/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/gtauzin/yohou)
![doc](https://github.com/gtauzin/yohou/actions/workflows/deploy-gh-pages.yml/badge.svg)
[![Downloads PyPI](https://static.pepy.tech/personalized-badge/hiclass?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=pypi)](https://pypi.org/project/hiclass/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**yohou** is a Python framework for time series forecasting based on
[scikit-learn](https://scikit-learn.org) and [polars](https://pola.rs/).

## Features

- **Scikit-learn Estimators:**
- **Plotly DataFrames:**
- **Optuna Samplers:**
- **Conformal Prediction-based Interval Forecasters:**
- **Plotly Interactive Visualisation tools:**

**Any feature missing on this list?** Search our [issue tracker](https://github.com/gtauzin/yohou/issues) to see if someone has already requested it and add a comment to it explaining your use-case. Otherwise, please open a new issue describing the requested feature and possible use-case scenario. We prioritize our roadmap based on user feedback, so we would love to hear from you.


## Roadmap

Here is our public roadmap: https://github.com/gtauzin/yohou/projects/1.

We do Just-In-Time planning, and we tend to reprioritize based on your feedback. Hence, items you see on this roadmap are subject to change. We prioritize features based on the number of people asking for it, features/fixes that are small enough and can be addressed while we work on other related features, features/fixes that help improve stability & relevance and features that address interesting use cases that excite us! If you would like to have a request prioritized, we ask that you add a detailed use-case for it, either as a comment on an existing issue (besides a thumbs-up) or in a new issue. The detailed context helps.


## Install

`yohou` and its dependencies can be easily installed with pip:

```shell
pip install yohou
```

## Quick start

Here's a quick example showcasing how you can train and predict using a local classifier per node, with a `RandomForestClassifier` for each node:

```python
from hiclass import LocalClassifierPerNode
from sklearn.ensemble import RandomForestClassifier

# Define data
X_train = [[1], [2], [3], [4]]
X_test = [[4], [3], [2], [1]]
Y_train = [
    ['Animal', 'Mammal', 'Sheep'],
    ['Animal', 'Mammal', 'Cow'],
    ['Animal', 'Reptile', 'Snake'],
    ['Animal', 'Reptile', 'Lizard'],
]

# Use random forest classifiers for every node
rf = RandomForestClassifier()
classifier = LocalClassifierPerNode(local_classifier=rf)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
```

HiClass can also be adopted in scikit-learn pipelines, and fully supports sparse matrices as input. In order to demonstrate the use of both of these features, we will use the following example:

```python
from hiclass import LocalClassifierPerParentNode
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Define data
X_train = [
    'Struggling to repay loan',
    'Unable to get annual report',
]
X_test = [
    'Unable to get annual report',
    'Struggling to repay loan',
]
Y_train = [
    ['Loan', 'Student loan'],
    ['Credit reporting', 'Reports']
]
```

Now, let's build a pipeline that will use `CountVectorizer` and `TfidfTransformer` to extract features as sparse matrices:

```python
# Use logistic regression classifiers for every parent node
lr = LogisticRegression()
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('lcppn', LocalClassifierPerParentNode(local_classifier=lr)),
])
```

Finally, let's train and predict with the pipeline we just created:

```python
# Train local classifier per parent node
pipeline.fit(X_train, Y_train)

# Predict
predictions = pipeline.predict(X_test)
```

## Explaining Hierarchical Classifiers

Hierarchical classifiers can provide additional insights when combined with explainability methods. HiClass allows explaining hierarchical models using SHAP values. Different hierarchical models yield different insights. More information on explaining [Local classifier per parent node](https://colab.research.google.com/drive/1rVlYuRU_uO1jw5sD6qo2HoCpCz6E6z5J?usp=sharing), [Local classifier per node](https://colab.research.google.com/drive/1wqSl1t_Qn2f62WNZQ48mdB0mNeu1XSF1?usp=sharing), and [Local classifier per level](https://colab.research.google.com/drive/1VnGlJu-1wSG4wxHXL0Ijf2a7Pu3kklT-?usp=sharing) is available on [Read the Docs](https://hiclass.readthedocs.io/en/latest/algorithms/explainer.html).

## Step-by-step walk-through

A step-by-step walk-through is available on our documentation
hosted on [Read the Docs](https://hiclass.readthedocs.io/en/latest/index.html).

This will guide you through the process of installing `yohou`
within a virtual environment, training, predicting, persisting models and much more.

## API documentation

Here's our official API documentation, available on [Read the Docs](https://hiclass.readthedocs.io/en/latest/api/index.html).

If you notice any issues with the documentation or walk-through,
please let us know by opening an issue here:
[https://github.com/gtauzin/yohou/issues](https://github.com/gtauzin/yohou/issues).

## FAQ

### How do the hierarchical classifiers work?

A detailed description on how the classifiers work is available at the [Algorithms Overview](https://hiclass.readthedocs.io/en/latest/algorithms/index.html) section on Read the Docs.

## Support

If you run into any problems or issues, please create a
[Github issue](https://github.com/gtauzin/yohou/issues) and we'll try our best to help.

We strive to provide good support through our issue tracker on Github.
However, if you'd like to receive private support with:

- Phone / video calls to discuss your specific use case and get recommendations
- Private discussions over Slack or Mattermost

Please reach out to fabio.malchermiranda@hpi.de.

## Contributing

We are a small team on a mission to democratize hierarchical classification, and we
will take all the help we can get! If you would like to get involved, here is
information on [contribution guidelines and how to test the code locally](https://github.com/gtauzin/yohou/blob/main/CONTRIBUTING.md).

You can contribute in multiple ways, e.g., reporting bugs, writing or
translating documentation, reviewing or refactoring code, requesting or
implementing new features, etc.

## Getting the latest updates

If you'd like to get updates when we release new versions, please click on the "Watch"
button on the top and select "Releases only". Github will then send you notifications
along with a changelog with each new release.

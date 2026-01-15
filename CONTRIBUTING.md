# Introduction

Thank you for considering contributing to Yohou! We welcome contributions in the form of pull requests, issues or code reviews. You can add to code, or simply send us spelling and grammar fixes or extra tests. Contribute anything that you think improves the community for us all!

The following sections describe our vision and the contribution process.

## Code of conduct

The Yohou team pledges to foster and maintain a welcoming and friendly community in all of our spaces. All members of our community are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md), and we will do our best to enforce those principles and build a happy environment where everyone is treated with respect and dignity.

## Get started

We use [GitHub Issues](https://github.com/gtauzin/yohou/issues) to keep track of known bugs. We keep a close eye on them and try to make it clear when we have an internal fix in progress. Before reporting a new issue, please do your best to ensure your problem hasn't already been reported. If so, it's often better to just leave a comment on an existing issue, rather than create a new one. Old issues also can often include helpful tips and solutions to common problems.

If you have already checked the [existing issues](https://github.com/gtauzin/yohou/issues) on GitHub and are still convinced that you have found odd or erroneous behaviour then please file a [new issue](https://github.com/gtauzin/yohou/issues/new/choose). We have a template that helps you provide the necessary information we'll need in order to address your query.

## Feature requests

### Suggest a new feature

If you have new ideas for Yohou functionality then please open a [GitHub issue](https://github.com/gtauzin/yohou/issues) with the label `enhancement`. Please describe in your own words the feature you would like to see, why you need it, and how it should work.

### Contribute a new feature

If you're unsure where to begin contributing to Yohou, please start by looking through the `good first issue` and `help wanted` on [GitHub](https://github.com/gtauzin/yohou/issues).

Typically, small contributions to `yohou` are more preferable due to an easier review process, but we accept any new features if they prove to be essential for the functioning of the package or if we believe that they are used by most projects.

## Your first contribution

Working on your first pull request? You can learn how from these resources:

* [First timers only](https://www.firsttimersonly.com/)
* [How to contribute to an open source project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)

### Guidelines

* Aim for cross-platform compatibility on Windows, macOS and Linux
* We use [uv](https://docs.astral.sh/uv/) for project and virtual environment management
* We use [SemVer](https://semver.org/) for versioning

Our code is designed to be compatible with Python 3.12 onwards and our style guidelines are (in cascading order):

* [PEP 8 conventions](https://www.python.org/dev/peps/pep-0008/) for all Python code
* [NumPy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for code comments (enforced at 100% coverage by `interrogate`)
* [PEP 484 type hints](https://www.python.org/dev/peps/pep-0484/) for all user-facing functions / class methods e.g.

```python
def count_truthy(elements: List[Any]) -> int:
    """Count truthy elements in a list.

    Parameters
    ----------
    elements : List[Any]
        List of elements to count.

    Returns
    -------
    int
        Number of truthy elements.

    """
    return sum(1 for elem in elements if elem)
```

> *Note:* We only accept contributions under the [BSD License](https://opensource.org/licenses/BSD-3-Clause) license and you should have permission to share the submitted code.

### Branching conventions

We use a branching model that helps us keep track of branches in a logical, consistent way. All branches should have the hyphen-separated convention of: `<type-of-change>/<short-description-of-change>` e.g. `feature/awesome-new-feature`

| Types of changes | Description                                                                 |
| ---------------- | --------------------------------------------------------------------------- |
| `docs`           | Changes to the documentation of the package                                 |
| `feature`        | Non-breaking change which adds functionality                                |
| `fix`            | Non-breaking change which fixes an issue                                    |
| `tests`          | Changes to project unit (`tests/`) and / or integration (`features/`) tests |

## Contribution process

 1. Fork the project
 2. Develop your contribution in a new branch.
 3. Make sure all your commits are signed off by using `-s` flag with `git commit`.
 4. Open a PR against the `main` branch and sure that the PR title follows the [Conventional Commits specs](https://www.conventionalcommits.org/en/v1.0.0/).
 5. Make sure the CI builds are green (have a look at the section [Testing](#testing) below)
 6. Ensure the documentation changes render properly (see section [Documentation](#documentation))
 7. Update the PR according to the reviewer's comments

## Documentation

* The main documentation is in the `docs/` directory and is built with [MkDocs](https://www.mkdocs.org/). To build or serve the documentation locally, use:

  ```bash
  uvx nox -s docs
  ```

  or

  ```bash
  uvx nox -s serve_docs
  ```

* API and CLI reference can be auto-generated from the code and docstrings but need to be manually added to the docs. Please keep docstrings up to date when contributing code.
* When adding new features or making changes, update the relevant documentation pages in `docs/pages/` and ensure the navigation in `mkdocs.yml` is correct.

## Testing

* **Unit tests** are in the `src/yohou/tests/` directory and use `pytest`. Run them with:

  ```bash
  uvx nox -s test
  ```

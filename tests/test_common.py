"""Tests based on the scikit-learn common tests."""


# @pytest.mark.skip(reason="Under development")
# # parametrize_with_checks allows to get a generator of check that is more fine-grained
# # than check_estimator
# @parametrize_with_checks(
#     [
#         est() for _, est in all_estimators()
#         if est.__name__ not in [
#             "BaseIntervalForecaster",
#             "BasePointForecaster",
#             "BaseWrapper",
#             "Sampler",
#             "Storage",
#             "ColumnTransformer",
#             "FeatureUnion",
#             "FeaturePipeline",
#             "OptunaSearchCV",
#         ]
#     ]
# )
# def test_estimators(estimator, check, request):
#     """Check the compatibility with scikit-learn API"""
#     check(estimator)

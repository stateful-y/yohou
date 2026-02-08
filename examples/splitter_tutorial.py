import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # ⏱️ Time Series Cross-Validation & Splitters

    In this tutorial, you will learn how to:
    - Perform temporal cross-validation correctly (no peeking into the future!)
        - Understand different splitter types: **Expanding Window** and **Sliding Window**
        - Learn how to use **gaps** to simulate deployment delays

    Traditional k-fold CV destroys temporal order. Yohou provides specific splitters to respect causality.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    from yohou.model_selection import (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
    )
    return (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
        go,
        make_subplots,
        np,
        pl,
    )


@app.cell
def _(pl):
    _url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    y = pl.read_csv(_url).select(
        pl.col("Month").str.strptime(pl.Datetime, "%Y-%m").alias("time"),
        pl.col("Passengers").cast(pl.Float64).alias("passengers"),
    )
    return (y,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 👁️ Visual Guide to Splitters

    Let's first visualize the conceptual difference between the main splitter strategies before applying them to data.
    """)
    return


@app.cell
def _(go, make_subplots):
    def plot_splitter_concept():
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Expanding Window",
                "Sliding Window",
                "Expanding Window with Gap",
                "Sliding Window with Gap"
            ),
            vertical_spacing=0.1
        )

        # Helper to add bars
        def add_split(row, step, train_start, train_len, test_start, test_len, gap_len=0):
            # Train
            fig.add_trace(go.Bar(
                x=[train_len],
                y=[f"Split {step}"],
                base=[train_start],
                orientation='h',
                marker_color='rgb(31, 119, 180)',
                name='Train' if step==1 and row==1 else None,
                showlegend=(step==1 and row==1)
            ), row=row, col=1)

            # Gap
            if gap_len > 0:
                fig.add_trace(go.Bar(
                    x=[gap_len],
                    y=[f"Split {step}"],
                    base=[train_start + train_len],
                    orientation='h',
                    marker_color='rgb(128, 128, 128)',
                    name='Gap' if step==1 and row==3 else None,
                    showlegend=(step==1 and row==3),
                    opacity=0.5
                ), row=row, col=1)

            # Test
            fig.add_trace(go.Bar(
                x=[test_len],
                y=[f"Split {step}"],
                base=[train_start + train_len + gap_len],
                orientation='h',
                marker_color='rgb(255, 127, 14)',
                name='Test' if step==1 and row==1 else None,
                showlegend=(step==1 and row==1)
            ), row=row, col=1)

        # 1. Expanding
        for i in range(1, 4):
            # args: row, step, train_start, train_len, test_start, test_len
            add_split(1, i, 0, 10 + i*2, 10 + i*2, 5)

        # 2. Sliding
        for i in range(1, 4):
             add_split(2, i, i*2, 10, i*2 + 10, 5)

        # 3. Expanding with Gap
        for i in range(1, 4):
            add_split(3, i, 0, 10 + i*2, 10 + i*2 + 2, 5, gap_len=2)

        # 4. Sliding with Gap
        for i in range(1, 4):
            add_split(4, i, i*2, 10, i*2 + 10 + 2, 5, gap_len=2)

        fig.update_layout(
            height=800,
            title_text="Splitter Strategies Visualization",
            barmode='stack',
            xaxis1=dict(range=[0, 30], showticklabels=False),
            xaxis2=dict(range=[0, 30], showticklabels=False),
            xaxis3=dict(range=[0, 30], showticklabels=False),
            xaxis4=dict(range=[0, 30], showticklabels=False),
            template="plotly_white"
        )
        return fig

    concept_fig = plot_splitter_concept()
    concept_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🎛️ Interactive Playground

    Below users can tweak parameters to see how splitters behave on the real Air Passengers dataset.
    """)
    return


@app.cell
def _(mo):
    splitter_type = mo.ui.dropdown(
        ["Expanding", "Sliding"], value="Expanding", label="Splitter Type"
    )
    n_splits = mo.ui.slider(2, 20, value=5, label="Number of Splits (n_splits)")
    test_size = mo.ui.slider(1, 48, value=12, label="Test Size (steps)")
    gap = mo.ui.slider(0, 12, value=0, label="Gap (steps)")
    stride = mo.ui.slider(1, 24, value=12, label="Stride (steps)")

    params = mo.md(
        f"""
        ### Settings
        {splitter_type}
        {n_splits}
        {test_size}
        {gap}
        {stride}
        
        *Note: 'Stride' only applies to Sliding Windows.*
        """
    )
    return gap, n_splits, params, splitter_type, stride, test_size


@app.cell
def _(
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
    gap,
    go,
    n_splits,
    splitter_type,
    stride,
    test_size,
    y,
):
    # Extract values
    _type = splitter_type.value
    _n_splits = n_splits.value
    _test_size = test_size.value
    _gap = gap.value
    _stride = stride.value

    # Instantiate splitter
    if _type == "Expanding":
        splitter = ExpandingWindowSplitter(
            n_splits=_n_splits,
            test_size=_test_size,
            gap=_gap if _gap > 0 else None
        )
    elif _type == "Sliding":
        # Fixed train_size for demo since UI doesn't provide it
        splitter = SlidingWindowSplitter(
            train_size=48,
            test_size=_test_size,
            stride=_stride,
            gap=_gap if _gap > 0 else None
        )

    # Calculate splits
    splits = list(splitter.split(y))

    # Create figure
    fig = go.Figure()

    # Iterate through splits
    for i, (train_idx, test_idx) in enumerate(splits):
        # Get timestamps
        t_train = y[train_idx]["time"]
        t_test = y[test_idx]["time"]

        # Plot Train
        fig.add_trace(
            go.Scatter(
                x=[t_train[0], t_train[-1]],
                y=[i, i],
                mode="lines",
                line=dict(color="blue", width=10),
                name="Train" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="x+y",
            )
        )

        # Plot Test
        fig.add_trace(
            go.Scatter(
                x=[t_test[0], t_test[-1]],
                y=[i, i],
                mode="lines",
                line=dict(color="orange", width=10),
                name="Test" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="x+y",
            )
        )

    fig.update_layout(
        title=f"Splitter Visualization: {_type}",
        xaxis_title="Time",
        yaxis_title="Fold",
        yaxis=dict(tickmode="linear", dtick=1),
        height=400 + (len(splits) * 20),
        template="plotly_white",
    )
    return fig, splitter


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What to look for:
    - **Overlap**: In Sliding Window, notice how the train sets shift but maintain fixed size.
    - **Gap**: If Gap is selected (and > 0), observe the empty space between blue (Train) and orange (Test).
    - **No Peeking**: The Test set (orange) is always legally *after* the Train set (blue).
    """)
    return


@app.cell
def _(go, splitter, y):
    # Retrieve sizes from the generator
    _train_sizes = []
    _test_sizes = []

    for _train_idx, _test_idx in splitter.split(y):
        _train_sizes.append(len(_train_idx))
        _test_sizes.append(len(_test_idx))

    # Create figure with secondary y-axis
    progression_fig = go.Figure()

    # Train Size (Left Axis)
    progression_fig.add_trace(
        go.Scatter(
            x=list(range(len(_train_sizes))),
            y=_train_sizes,
            name="Train Size",
            mode="lines+markers",
            line=dict(color="rgb(31, 119, 180)"),
        )
    )

    # Test Size (Right Axis)
    progression_fig.add_trace(
        go.Scatter(
            x=list(range(len(_test_sizes))),
            y=_test_sizes,
            name="Test Size",
            mode="lines+markers",
            line=dict(color="orange"),
            yaxis="y2",
        )
    )

    progression_fig.update_layout(
        title="Train vs Test Size Progression",
        xaxis_title="Fold Index",
        yaxis=dict(
            title=dict(text="Train Sample Size", font=dict(color="rgb(31, 119, 180)")),
            tickfont=dict(color="rgb(31, 119, 180)"),
        ),
        yaxis2=dict(
            title=dict(text="Test Sample Size", font=dict(color="orange")),
            tickfont=dict(color="orange"),
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
    )
    progression_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 💻 Manual Cross-Validation Loop

    While `RandomizedSearchCV` is powerful, sometimes you need manual control. The following example shows how to iterate through splits, fit a model, and evaluate performance manually.
    """)
    return


@app.cell
def _(pl, splitter, y):
    from yohou.metrics.point import MeanAbsoluteError
    from yohou.point_forecaster.naive import SeasonalNaive

    _forecaster = SeasonalNaive(seasonality=12)
    _scorer = MeanAbsoluteError()
    _results = []

    for _i, (_train_idx, _test_idx) in enumerate(splitter.split(y)):
        _y_train = y[_train_idx]
        _y_test = y[_test_idx]
        _fh = len(_y_test)

        _forecaster.fit(_y_train, forecasting_horizon=_fh)
        _y_pred = _forecaster.predict(forecasting_horizon=_fh)

        _mae = _scorer.score(_y_test, _y_pred)

        _results.append({
            "fold": _i,
            "train_len": len(_train_idx),
            "mae": _mae
        })

    results_df = pl.DataFrame(_results)
    results_df
    return MeanAbsoluteError, SeasonalNaive


@app.cell
def _(mo):
    mo.md(r"""
    ## 🔍 Hyperparameter Tuning with RandomizedSearchCV

    Just like sklearn's `GridSearchCV` or `RandomizedSearchCV`, Yohou provides time series-aware search classes that respect time series boundaries during cross-validation.
    """)
    return


@app.cell
def _(MeanAbsoluteError, SeasonalNaive, splitter, y):
    from scipy.stats import randint

    from yohou.model_selection import RandomizedSearchCV

    # Setup RandomizedSearchCV
    search = RandomizedSearchCV(
        forecaster=SeasonalNaive(),
        param_distributions={
            "seasonality": randint(3, 25)  # Random integers from 3 to 24
        },
        cv=splitter,
        scoring=MeanAbsoluteError(),
        n_iter=10,
        verbose=1,
    )

    # Fit
    search.fit(y, forecasting_horizon=12)

    # Display results
    print(f"Best params: {search.best_params_}")
    print(f"Best score: {search.best_score_}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 📊 Splitter Performance Comparison

    Different splitting strategies can yield different performance estimates. Below we compare the stability (standard deviation of MAE) and performance (mean MAE) across different splitter types.
    """)
    return


@app.cell
def _(
    ExpandingWindowSplitter,
    MeanAbsoluteError,
    SeasonalNaive,
    SlidingWindowSplitter,
    np,
    pl,
    y,
):
    def evaluate_splitter(splitter_cls, **kwargs):
        _splitter = splitter_cls(**kwargs)
        _forecaster = SeasonalNaive(seasonality=12)
        _scorer = MeanAbsoluteError()
        _scores = []

        for _train_idx, _test_idx in _splitter.split(y):
            _y_train = y[_train_idx]
            _y_test = y[_test_idx]
            _fh = len(_y_test)

            _forecaster.fit(_y_train, forecasting_horizon=_fh)
            _y_pred = _forecaster.predict(forecasting_horizon=_fh)
            _scores.append(_scorer.score(_y_test, _y_pred))

        return {
            "mean_mae": np.mean(_scores),
            "std_mae": np.std(_scores)
        }

    # Compare strategies
    strategies = [
        ("Expanding", ExpandingWindowSplitter, {"n_splits": 5, "test_size": 12}),
        ("Expanding+Gap", ExpandingWindowSplitter, {"n_splits": 5, "test_size": 12, "gap": 2}),
        ("Sliding", SlidingWindowSplitter, {"train_size": 48, "test_size": 12, "stride": 12}),
    ]

    results = []
    for name, cls, kwargs in strategies:
        try:
            stats = evaluate_splitter(cls, **kwargs)
            results.append({"Strategy": name, **stats})
        except Exception:
             results.append({"Strategy": name, "mean_mae": None, "std_mae": None})

    comparison_df = pl.DataFrame(results)
    comparison_df
    return (strategies,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 🏢 Handling Panel Data

    Yohou splitters are panel-aware but operate simply: **all groups are split together based on the time index**. If your data has multiple groups, the splitter ensures that the 'cut' happens at the same time point for every group.
    """)
    return


@app.cell
def _(ExpandingWindowSplitter, go, np, pl):
    from datetime import datetime

    # Mock panel data
    dates = pl.datetime_range(start=datetime(2023,1,1), end=datetime(2023,6,1), interval="1mo", eager=True)
    panel_df = pl.DataFrame({
        "time": dates,
        "sales__store_A": np.random.rand(len(dates)) * 100,
        "sales__store_B": np.random.rand(len(dates)) * 100 + 50
    })

    # Split
    panel_splitter = ExpandingWindowSplitter(n_splits=2, test_size=1)

    # Get first split
    panel_train_idx, panel_test_idx = next(panel_splitter.split(panel_df))

    train = panel_df[panel_train_idx]
    test = panel_df[panel_test_idx]

    # Visualize
    panel_fig = go.Figure()

    for col, color in [("sales__store_A", "blue"), ("sales__store_B", "green")]:
        # Train
        panel_fig.add_trace(go.Scatter(
            x=train["time"], y=train[col],
            mode='lines+markers', name=f"Train {col}",
            line=dict(color=color)
        ))
        # Test
        panel_fig.add_trace(go.Scatter(
            x=test["time"], y=test[col],
            mode='lines+markers', name=f"Test {col}",
            line=dict(color=color, dash='dot')
        ))

    panel_fig.update_layout(title="Panel Data Split (Same Cut-off for All Groups)")
    panel_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🔥 Error Heatmap

    The heatmap below displays the Mean Absolute Error (MAE) for each fold across different splitter strategies. This visualization helps identify if certain time periods (folds) are consistently more difficult to forecast, regardless of the splitting strategy used.
    """)
    return


@app.cell
def _(MeanAbsoluteError, SeasonalNaive, go, strategies, y):
    # Calculate scores per fold for each strategy
    _scores_matrix = []
    _strategy_names = []
    _max_folds = 0

    # Iterate through strategies defined in the previous cell
    for _name, _cls, _kwargs in strategies:
        _strategy_names.append(_name)
        _splitter = _cls(**_kwargs)

        _current_scores = []
        _forecaster = SeasonalNaive(seasonality=12)
        _scorer = MeanAbsoluteError()

        for _train_idx, _test_idx in _splitter.split(y):
            _y_train = y[_train_idx]
            _y_test = y[_test_idx]
            _fh = len(_y_test)

            _forecaster.fit(_y_train, forecasting_horizon=_fh)
            _y_pred = _forecaster.predict(forecasting_horizon=_fh)
            _current_scores.append(_scorer.score(_y_test, _y_pred))

        _scores_matrix.append(_current_scores)
        _max_folds = max(_max_folds, len(_current_scores))

    # Pad with None for uneven fold counts and transpose for the heatmap
    # We want X-axis = Strategies, Y-axis = Folds
    _transposed_scores = []
    for _i in range(_max_folds):
        _fold_scores = []
        for _s_scores in _scores_matrix:
            if _i < len(_s_scores):
                _fold_scores.append(_s_scores[_i])
            else:
                _fold_scores.append(None)
        _transposed_scores.append(_fold_scores)

    # Create Heatmap
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=_transposed_scores,
            x=_strategy_names,
            y=[f"Fold {_i}" for _i in range(_max_folds)],
            colorscale="Viridis",
            colorbar=dict(title="MAE"),
            text=[
                [f"{_val:.2f}" if _val is not None else "" for _val in _row]
                for _row in _transposed_scores
            ],
            texttemplate="%{text}",
            xgap=1,
            ygap=1,
        )
    )

    heatmap_fig.update_layout(
        title="MAE Heatmap by Fold and Strategy",
        xaxis_title="Splitter Strategy",
        yaxis_title="Fold Index",
        template="plotly_white",
        height=500,
    )
    heatmap_fig
    return


if __name__ == "__main__":
    app.run()

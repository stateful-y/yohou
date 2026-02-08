import marimo as mo

__generated_with = "0.10.9"
app = mo.App(width="medium")

@app.cell
def mo():
    import marimo as mo
    return mo,

@app.cell
def imports():
    import polars as pl
    from yohou import datasets
    return datasets, pl

@app.cell
def inspection_helper(mo, pl):
    def inspect(df: pl.DataFrame, name: str):
        mo.output.append(mo.md(f"### {name}"))
        mo.output.append(mo.md(f"**Shape**: {df.shape}"))
        mo.output.append(mo.md("**Schema**:"))
        mo.output.append(df.schema)
        mo.output.append(mo.md("**First 5 rows**:"))
        mo.output.append(df.head())
        
        # Simple plot if "y" exists
        if "y" in df.columns and "time" in df.columns:
            # For panel data, just plot first group or aggregated
            try:
                # Use plotly if available or simple marimo default
                # mo.ui.table is nice
                pass
            except:
                pass
    return inspect,

@app.cell
def load_air(datasets, inspect):
    df = datasets.load_air_passengers()
    inspect(df, "Air Passengers")
    return df,

@app.cell
def load_sunspots(datasets, inspect):
    df = datasets.load_sunspots()
    inspect(df, "Sunspots")
    return df,

@app.cell
def load_m4(datasets, inspect):
    df = datasets.load_m4_monthly()
    inspect(df, "M4 Monthly (Subset)")
    return df,

@app.cell
def load_tourism(datasets, inspect):
    df = datasets.load_australian_tourism()
    inspect(df, "Australian Tourism")
    return df,

@app.cell
def load_electricity(datasets, inspect):
    df = datasets.load_vic_electricity()
    inspect(df, "Victoria Electricity")
    return df,

@app.cell
def load_store(datasets, inspect):
    df = datasets.load_store_sales()
    inspect(df, "Store Sales")
    return df,

@app.cell
def load_walmart(datasets, inspect):
    df = datasets.load_walmart_sales()
    inspect(df, "Walmart Sales")
    return df,

@app.cell
def load_ett(datasets, inspect):
    df = datasets.load_ett_m1()
    inspect(df, "ETTm1")
    return df,

if __name__ == "__main__":
    app.run()

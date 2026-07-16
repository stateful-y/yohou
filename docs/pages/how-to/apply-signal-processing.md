# How to Apply Signal Processing

This guide shows you how to smooth time series with digital filters, extract
rate-of-change features, and use frequency-domain plots to choose filter
parameters.

## Prerequisites

- Familiarity with transformers ([How to Use Preprocessing Transformers](use-preprocessing-transformers.md))
- Understanding of the fit/transform pattern ([Getting Started](../tutorials/getting-started.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Smooth High-Frequency Noise

[`NumericalFilter`](/pages/api/generated/yohou.preprocessing.NumericalFilter/)
applies standard digital IIR filters. A lowpass filter removes noise above a
chosen cutoff frequency:

```python
from yohou.preprocessing import NumericalFilter

smoother = NumericalFilter(
    design="butterworth",
    mode="lowpass",
    order=4,
    cutoff_frequency=0.2,
)
df_smooth = smoother.fit_transform(df)
```

The `cutoff_frequency` is normalized between 0 and 1, where 1 is the Nyquist
frequency (half the sampling rate). Use [`plot_spectrum`](/pages/api/generated/yohou.plotting.plot_spectrum/)
(below) to identify where noise begins and set the cutoff just below that point.

Start with `design="butterworth"` and `order=4`. If the filtered signal still
contains unwanted frequencies, increase the order for a sharper rolloff. If you
need an even steeper cutoff, switch to `"chebyshev1"` for passband ripple
tolerance (sharper rolloff) or `"chebyshev2"` for stopband attenuation (cleaner
rejection band). Use `"elliptic"` for the sharpest possible cutoff when both
passband and stopband ripple are acceptable. Use `"bessel"` when preserving
waveform shape matters more than a sharp cutoff.

## Remove Low-Frequency Drift

Use a highpass filter to remove slow trends or baseline drift while preserving
faster dynamics:

```python
detrend = NumericalFilter(
    design="butterworth",
    mode="highpass",
    order=4,
    cutoff_frequency=0.05,
)
df_detrended = detrend.fit_transform(df)
```

## Isolate a Frequency Band

For bandpass or bandstop filtering, pass a 2-tuple as the `cutoff_frequency`:

```python
# Keep only frequencies between 0.1 and 0.4 of Nyquist
bandpass = NumericalFilter(
    mode="bandpass",
    cutoff_frequency=(0.1, 0.4),
)
df_band = bandpass.fit_transform(df)
```

Use `mode="bandstop"` instead to remove a specific frequency band (e.g., a
known interference frequency).

## Extract Rate of Change

[`NumericalDifferentiator`](/pages/api/generated/yohou.preprocessing.NumericalDifferentiator/)
computes numerical derivatives using central differences:

```python
from yohou.preprocessing import NumericalDifferentiator

diff = NumericalDifferentiator(order=1)
df_rate = diff.fit_transform(df)
```

The `order` parameter controls boundary accuracy: `1` uses 2-point differences
at boundaries, `2` uses 3-point second-order accurate differences.

To integrate a derivative back, use
[`NumericalIntegrator`](/pages/api/generated/yohou.preprocessing.NumericalIntegrator/):

```python
from yohou.preprocessing import NumericalIntegrator

integrator = NumericalIntegrator(method="cumulative_trapezoid")
df_integrated = integrator.fit_transform(df)
```

The `method` parameter accepts `"cumulative_trapezoid"` (faster) or
`"cumulative_simpson"` (more accurate).

## Inspect the Frequency Spectrum

Use [`plot_spectrum`](/pages/api/generated/yohou.plotting.plot_spectrum/)
to visualize the power spectral density before choosing a cutoff:

```python
from yohou.plotting import plot_spectrum

fig = plot_spectrum(df, columns="value", show_peaks=True, n_peaks=3)
fig.show()
```

The `show_peaks` option annotates the dominant frequencies with their
corresponding period in sample units, making it easier to identify the
boundary between signal and noise.

## Check Phase Alignment

Use [`plot_phase`](/pages/api/generated/yohou.plotting.plot_phase/)
to inspect the phase angle of each frequency component via FFT:

```python
from yohou.plotting import plot_phase

fig = plot_phase(df, columns="value")
fig.show()
```

Compare the phase plot before and after filtering to verify that the filter
does not introduce unacceptable phase distortion at frequencies you care about.

## See Also

- [Preprocessing](../explanation/preprocessing.md) for the conceptual background on transformers
- [Visualization](../explanation/visualization.md) for other plotting functions

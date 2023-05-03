# Neurocodec
Closed-Loop Neurostimulation with Delay Compensation

## Prerequisites
Install required python libraries with `pip install -r requirements.txt`.

## Usage
Create a `StateSpace` system with `model.py`:
* `neural_oscillator`: simple linear brain model.
* `cortico_thalamic`: non-linear brain model with cortico-thalamic loop.
* `alpha_gamma_filter`: weighted double bandpass filter with a positive weight in the alpha-band and a negative weight in the gamma-band.
**NB**: The first input of the brain systems is the stimulation input while all the other inputs are the driving noise with standard normal distribution.

Build closed-loop circuits using the `build.py` module.

Run numerical simulations using the `sim.py` module.

Analyse simulated time-series using the `eeg.py` module.

Plot time-series and spectra using the `plot.py` module.

## Examples
In `main.py`, uncomment a given function `figure.<f>()` and run the file.
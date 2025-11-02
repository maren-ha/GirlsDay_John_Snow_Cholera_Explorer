# 🦠 Epidemiological Data Explorer

## Context

This app was developed as part of **SmallData’s Girls’ Day initiative** to promote female participation in STEM.  
It was designed and led by early-career researchers Maria Krissmer, Hannah Habenicht, Masako Kaufmann, and Maren Hackenberg.  
In this full-day program, eight high-school girls (ages 12–15) explored how data science can be used in biomedicine, culminating in a **“Disease Outbreak Challenge”** where they investigated a simulated epidemic using real-world analysis techniques.

## The challenge

**Outbreak in Soho (London), 1854.**  
An infectious disease with symptoms such as diarrhea, nausea, vomiting, and severe fluid loss is spreading fast. Within a few days, hundreds fall ill; many die. The cause is unknown, many believe in “bad air.” Fear, rumors, and confusion spread.

**Your role.**  
You step into the shoes of scientists and, using data, models, and experiments, try to find out:  
- How exactly does the disease spread?  
- Who is particularly at risk?  
- How can we stop it?

There are four tasks in the investigation: Population dynamics simulation, Analyze genome strains, Antibody studies, and **Epidemiological data analysis** (this repo).

## Exploratory flow used with students

1) **Distributions (stacked histogram/bar)** Distribution of potential risk factors, highlighting health status. For continuous variables, **bin width matters**.  
2) **Heatmap (compare two variables)** Direct comparison of two variables; continuous variables are **binned**.  
3) **Scatter (no binning)** More fine-grained view: two continuous variables as axes, **color-coded by disease status**. Use filters to see patterns strengthen.

**From visuals → hypothesis:** What pattern looks most plausible?  
**Confirm with statistics:** keep in mind we can test correlation, not causation. Then do simple tests/models (chi-squared, t-test, logistic with an interaction). Keep **confounding** in mind (e.g., a superspreader event near one pump could mimic a distance effect).

## For Educators

The repository is designed to support exploratory data-driven learning and discovery in introductory data science or biology settings.

- The **Streamlit app** provides an interactive exploration environment for students.
- The **worksheet** (Markdown) guides group discussions and helps structure the learning process.
- The dataset is **synthetic**, allowing safe experimentation while illustrating real-world reasoning challenges in epidemiology. It can be re-generated or modified with the script in `src/generate_data.py`

## Inspiration

This activity is inspired by **John Snow’s 1854 Cholera Outbreak investigation**, a landmark case in the history of epidemiology and data visualization.  
In this educational version, the scenario and data have been **simulated and simplified** to make the core reasoning process accessible to students.  
Participants explore, hypothesize, and test ideas about how the disease spreads, mirroring how Snow combined maps, data, and logic to uncover hidden patterns.

## Repository Overview

- `app`: contains the `streamlit_app.py` interactive app 
- `data`: contains the synthetic dataset `cholera_dataset.csv`
- `materials`: contains a `Worksheet.md`
- `scripts`: install helpers using `conda` and `pip` (`setup-conda.sh` and `setup-venv.sh`)
- `src`: source code: `explore_data.py` contains the code for writing the widgets that the app is based on; `generate_data.py` is the script for generating (and modifying) the synthetic data
- `environment.yml`: package dependencies; for installing the environment using `conda`
- `requirements.txt`: package dependencies; for installing the environment using `pip`
- `LICENSE.md`: MIT license 
- `README.md`: this Readme. 


## Installation (automated)

### Option A: Conda (recommended on macOS/Apple Silicon)
```bash
conda env create -f environment.yml    # first time
# or: conda env update -f environment.yml --prune
conda activate cholera-demo
```

### Option B: Pip virtualenv
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Shortcuts
- `make app` → run the Streamlit app
- `make notebook` → open Jupyter Lab
- Or run helper scripts:
  - `./scripts/setup-conda.sh`
  - `./scripts/setup-venv.sh`

# 🦠 Epidemiological Data Explorer

## Context

This app was developed as part of **SmallData’s Girls’ Day initiative** to promote female participation in STEM.  
It was designed and led by early-career researchers Maria Krissmer, Hannah Habenicht, Masako Kaufmann, and Maren Hackenberg.  
In this full-day program, eight high-school girls (ages 12–15) explored how data science can be used in biomedicine, culminating in a **“Disease Outbreak Challenge”** where they investigated a simulated epidemic using real-world analysis techniques.

## The challenge

**Outbreak in Soho (London), 1854.**  
An infectious disease with symptoms such as diarrhea, nausea, vomiting, and severe fluid loss is spreading fast. Within a few days, hundreds fall ill; many die. The cause is unknown, many believe in “bad air.” Fear, rumors, and confusion spread.

**Your role.**  
Imagine you are a scientist in 1854. You have gathered observations about ages, occupations, how many people live in each household, diet, health outcomes, and distances to water pumps. Now you need to look for patterns carefully:  
- Who seems most affected?  
- Which clues keep appearing?  
- Which hypothesis can you test with the data?

## Exploratory flow used with students

1) **Distributions (stacked histogram/bar)** Distribution of potential risk factors, highlighting health status. For numeric variables, the slider changes the **number of groups**.  
2) **Scatter (with light jitter)** More fine-grained view: choose numeric or categorical variables as axes, **color-coded by disease status**. Use filters to see patterns strengthen.

**From visuals → hypothesis:** What pattern looks most plausible?  
**Check with statistics:** use a simple **chi-squared test** for illness vs. nearest pump and a **logistic regression** using age and distance. Treat the numbers as clues that help you decide whether your idea deserves a closer look.

## For Educators

The repository is designed to support exploratory data-driven learning and discovery in introductory data science or biology settings.

- The **Streamlit app** provides an interactive exploration environment for students.
- The **worksheet** (Markdown) guides group discussions and helps structure the learning process.
- The dataset is **synthetic**, allowing safe experimentation while illustrating real-world reasoning challenges in epidemiology.
- The repository ships with both English and German CSVs in `data/`.
- The Streamlit app normalizes English and German column names internally, so either dataset can be loaded through the same interface.
- The data can be re-generated or modified with `src/generate_data.py` and `src/generate_data_de.py`.

## Inspiration

This activity is inspired by **John Snow’s 1854 Cholera Outbreak investigation**, a landmark case in the history of epidemiology and data visualization.  
In this educational version, the scenario and data have been **simulated and simplified** to make the core reasoning process accessible to students.  
Participants explore, hypothesize, and test ideas about how the disease spreads, mirroring how Snow combined maps, data, and logic to uncover hidden patterns.

## Repository Overview

- `app`: contains the `streamlit_app.py` interactive app 
- `data`: contains the synthetic datasets `cholera_dataset.csv` and `cholera_datensatz_de.csv`
- `materials`: contains a `Worksheet.md`
- `scripts`: install helpers using `conda` and `pip` (`setup-conda.sh` and `setup-venv.sh`)
- `src`: source code: `explore_data.py` and `explore_data_de.py` contain the notebook/widget explorers; `generate_data.py` and `generate_data_de.py` generate the synthetic datasets
- `environment.yml`: package dependencies; for installing the environment using `conda`
- `requirements.txt`: package dependencies; for installing the environment using `pip`
- `LICENSE`: MIT license 
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
- `make qr` → run the Streamlit app for devices on the same Wi-Fi and save a QR code to `assets/app-qr.png`
- `make notebook` → open Jupyter Lab
- Or run helper scripts:
  - `./scripts/setup-conda.sh`
  - `./scripts/setup-venv.sh`

## Same-Wi-Fi QR Code

For the simplest classroom setup, keep the teacher computer and student phones on the same Wi-Fi network, then run:

```bash
make qr
```

The script starts Streamlit on the local network, prints a URL like `http://192.168.x.x:8501`, and saves a QR code image at `assets/app-qr.png`. Students can scan that QR code and open the app in a browser while the terminal window stays open.

If phones cannot open the page, check that they are on the same Wi-Fi and that the computer firewall allows incoming connections for Python/Streamlit.

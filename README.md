# 🦠 Epidemiological Data Explorer — London 1854

## Context

This app was developed as part of **SmallData’s Girls’ Day initiative** to promote female participation in STEM.  
It was designed and led by early-career researchers **Maria Krissmer, Hannah Habenicht, Masako Kaufmann**, and **Maren Hackenberg**.  
In this full-day program, eight high-school girls (ages 12–15) explored how data science can be used in biomedicine, culminating in a **“Disease Outbreak Challenge”** where they investigated a simulated epidemic using real-world analysis techniques.

## The challenge

**Outbreak in Soho (London), 1854.**  
An infectious disease with symptoms such as diarrhea, nausea, vomiting, and severe fluid loss is spreading fast. Within a few days, hundreds fall ill; many die. The cause is unknown — many believe in “bad air.” Fear, rumors, and confusion spread.

**Your role.**  
You step into the shoes of scientists and, using data, models, and experiments, try to find out:  
- How exactly does the disease spread?  
- Who is particularly at risk?  
- How can we stop it?

There are four tasks in the investigation: Population dynamics simulation, Analyze genome strains, Antibody studies, and **Epidemiological data analysis** (this repo).

## Exploratory flow used with students

1) **Distributions (stacked histogram/bar)** — distribution of potential risk factors, highlighting health status. For continuous variables, **bin width matters**.  
2) **Heatmap (compare two variables)** — direct comparison of two variables; continuous variables are **binned**.  
3) **Scatter (no binning)** — fine-grained view: two continuous variables as axes, **color-coded by disease status**. Use filters to see patterns strengthen.

**From visuals → hypothesis:** What pattern looks most plausible?  
**Confirm with stats:** we can test **correlation**, not **causation**. Then do simple tests/models (chi-squared, t-test, logistic with an Age×Distance interaction). Keep **confounding** in mind (e.g., a superspreader event near one pump could mimic a distance effect).

## For Educators

The repository is designed to support **inquiry-based learning** in introductory data science or biology settings.

- The **Streamlit app** provides an interactive exploration environment for students.
- The **worksheet** (Markdown) guides group discussions and helps structure the learning process.
- The **teacher notebook** (`explore_data_teacher.ipynb`) includes Python code, explanations, and plots that parallel the app — ideal for walkthroughs or teacher prep.
- The dataset is **synthetic**, allowing safe experimentation while illustrating real-world reasoning challenges in epidemiology.

## Inspiration

This activity is inspired by **John Snow’s 1854 Cholera Outbreak investigation**, a landmark case in the history of epidemiology and data visualization.  
In this educational version, the scenario and data have been **simulated and simplified** to make the core reasoning process accessible to students.  
Participants explore, hypothesize, and test ideas about how the disease spreads, mirroring how Snow combined maps, data, and logic to uncover hidden patterns.

## Repository Overview

john-snow-cholera-demo/
│
├── app/
│ ├── streamlit_app.py # Interactive Streamlit app for students
│ ├── worksheet.md # Guided worksheet for classroom use
│ └── sample_data/ (optional) # Example datasets if included
│
├── notebooks/
│ └── explore_data_teacher.ipynb # Teacher-facing notebook version (same logic, with explanations)
│
├── data/
│ └── cholera_dataset_en.csv # Synthetic teaching dataset
│
├── README.md # Project documentation and educational context
└── requirements.txt # Python dependencies

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
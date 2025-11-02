
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Epidemiological Data Explorer — London 1854", layout="wide")

# ---------- Intro (full English context) ----------
st.title("🦠 Epidemiological Data Explorer — London 1854")

st.markdown(r"""
**Outbreak in London (1854).**  
An infectious disease with symptoms such as diarrhea, nausea, vomiting, and severe fluid loss. Within a few days, hundreds fall ill; many die. The cause is unknown — many believe in “bad air.” Fear, rumors, and confusion spread.

**Your role.**  
You step into the shoes of scientists and, using data, models, and experiments, try to find out:  
- How exactly does the disease spread?  
- Who is particularly at risk?  
- How can we stop it?

Four tasks in the investigation: **Population dynamics simulation**, **Analyze genome strains**, **Antibody studies**, and **Epidemiological data analysis** (this app).
""")

st.info("This app focuses on the *Epidemiological data analysis* track. Start with exploratory visuals, form a hypothesis, then use simple statistical checks to test for correlation (not causation).")

# ---------- Data ----------
@st.cache_data
def load_data():
    for path in ["data/cholera_dataset.csv","data/cholera_datensatz_de.csv","data/cholera_dataset_en.csv"]:
        try:
            return pd.read_csv(path), path
        except Exception:
            pass
    st.error("Could not find a dataset (expected e.g. data/cholera_dataset_en.csv)."); st.stop()

df, data_path = load_data()

# Harmonize columns
#rename_map = {
#    "Gender": "Gender", "Geschlecht": "Gender",
#    "Age": "Age", "Alter": "Age",
#    "Occupation": "Occupation", "Beruf": "Occupation",
#    "HouseholdSize": "Household Size", "Haushaltsgröße": "Household Size",
#    "RawVegConsumption": "Raw Vegetables", "Rohkost-Konsum": "Raw Vegetables",
#    "NearestPump": "Nearest Pump", "Nächstgelegene Pumpe": "Nearest Pump",
#    "HomeX": "Home Location X", "Wohnort X": "Home Location X",
#    "HomeY": "Home Location Y", "Wohnort Y": "Home Location Y",
#    "HealthStatus": "Health Status", "Gesundheitsstatus": "Health Status",
#    "DistanceToPumpA": "Distance to Pump A",
#    "DistanceToPumpB": "Distance to Pump B",
#    "DistanceToPumpC": "Distance to Pump C",
#    "DistanceToPumpD": "Distance to Pump D",
#}
#df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# Derived feature: household size category
def categorize_household_size(_df):
    if "Household Size" not in _df.columns: return _df
    _df = _df.copy()
    conds = [(_df["Household Size"] <= 2), (_df["Household Size"] <= 4), (_df["Household Size"] <= 6), (_df["Household Size"] > 6)]
    _df["Household Size Category"] = np.select(conds, ["1-2","3-4","5-6","7+"], default="Unknown")
    return _df

df = categorize_household_size(df)

# ---------- Sidebar Controls ----------
st.sidebar.header("Filters")
age_min = int(np.nanmin(df["Age"])) if "Age" in df.columns else 0
age_max = int(np.nanmax(df["Age"])) if "Age" in df.columns else 80
age_range = st.sidebar.slider("Age range", 0, max(80, age_max), (age_min, age_max), 1)

gender = st.sidebar.selectbox("Gender", ["All"] + (sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else []), index=0)
occupation = st.sidebar.selectbox("Occupation", ["All"] + (sorted(df["Occupation"].dropna().unique().tolist()) if "Occupation" in df.columns else []), index=0)
hh_cat = st.sidebar.selectbox("Household size", ["All"] + (["1-2","3-4","5-6","7+"] if "Household Size Category" in df.columns else []), index=0)
rv_col = "Raw Vegetables" if "Raw Vegetables" in df.columns else None
raw_veg = st.sidebar.selectbox("Raw vegetables", ["All"] + (sorted(df[rv_col].dropna().unique().tolist()) if rv_col else []), index=0) if rv_col else "All"
nearest_pump = st.sidebar.selectbox("Nearest pump", ["All","Pump A","Pump B","Pump C","Pump D"], index=0)
bins = st.sidebar.slider("Bin width (numeric variables)", 2, 20, 6, 1)

def apply_filters(_df):
    d = _df.copy()
    if "Age" in d.columns: d = d[(d["Age"] >= age_range[0]) & (d["Age"] <= age_range[1])]
    if gender != "All" and "Gender" in d.columns: d = d[d["Gender"] == gender]
    if occupation != "All" and "Occupation" in d.columns: d = d[d["Occupation"] == occupation]
    if hh_cat != "All" and "Household Size Category" in d.columns: d = d[d["Household Size Category"] == hh_cat]
    if raw_veg != "All" and rv_col: d = d[d[rv_col] == raw_veg]
    if nearest_pump != "All" and "Nearest Pump" in d.columns: d = d[d["Nearest Pump"] == nearest_pump]
    return d

# ---------- Tabs ----------
tab_intro, tab_dist, tab_heat, tab_scatter, tab_stats = st.tabs(
    ["Overview", "Distributions (stacked histogram)", "Heatmap (compare two variables)", "Scatter (no binning)", "Hypothesis & stats"]
)

# ---------- Overview ----------
with tab_intro:
    st.subheader("A quick look at the data")
    c1, c2 = st.columns([2,1])
    with c1:
        st.dataframe(apply_filters(df).head(20))
    with c2:
        st.metric("Rows", len(df)); st.metric("Columns", len(df.columns))
        if "Health Status" in df.columns:
            st.write("Health outcome (counts)")
            st.write(df["Health Status"].value_counts(dropna=False))
    st.markdown("**Missing values** are common in real datasets. Keep them in mind when interpreting any plot.")

    miss = df.isna().sum().sort_values(ascending=False)
    if miss.sum() > 0:
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(miss.index, miss.values)
        ax.set_xticklabels(miss.index, rotation=45, ha="right"); ax.set_ylabel("Missing")
        st.pyplot(fig)

# ---------- Distributions (stacked histogram / bar) ----------
with tab_dist:
    st.subheader("Distribution of potential risk factors — highlight health status")
    st.caption("Select a variable. If it’s continuous, we bin it — so **bin width matters**. This is exploratory.")
    exclude = {"ID","Home Location X","Home Location Y"}
    candidates = [c for c in df.columns if c not in exclude]
    var = st.selectbox("Variable", candidates, index=(candidates.index("Age") if "Age" in candidates else 0))

    d = df.copy()
    if var not in d.columns or "Health Status" not in d.columns:
        st.info("Need the selected variable and Health Status to draw this plot.")
    else:
        fig, ax = plt.subplots(figsize=(9,5))
        if pd.api.types.is_numeric_dtype(d[var]):
            groups = [g[var].dropna().values for _, g in d.groupby("Health Status")]
            labels = [str(s) for s, _ in d.groupby("Health Status")]
            ax.hist(groups, bins=bins, stacked=True, label=labels)
            ax.set_xlabel(var); ax.set_ylabel("Count")
        else:
            grouped = d.groupby([var, "Health Status"]).size().unstack(fill_value=0)
            grouped.plot(kind="bar", stacked=True, ax=ax)
            ax.set_xlabel(var); ax.set_ylabel("Count")
        ax.set_title("Stacked distribution by Health Status")
        ax.legend(title="Health Status")
        st.pyplot(fig)

    st.markdown("""
**Discuss:**  
- Which groups/bins look more affected?  
- Could this be driven by another variable (confounding)?  
- How does changing the **bin width** change your impression for continuous variables?
""")

# ---------- Heatmap (compare two variables, with binning for continuous) ----------
with tab_heat:
    st.subheader("Directly compare two variables (with binning for continuous)")
    options = [c for c in df.columns if c not in exclude and c != "Health Status"]
    if len(options) < 2:
        st.info("Need at least two variables to compare.")
    else:
        x = st.selectbox("X", options, index=0)
        y = st.selectbox("Y", options, index=1)
        d = df.dropna(subset=[x,y]).copy()
        if pd.api.types.is_numeric_dtype(d[x]):
            d[x] = pd.cut(pd.to_numeric(d[x], errors="coerce"), bins=bins, duplicates="drop")
        if pd.api.types.is_numeric_dtype(d[y]):
            d[y] = pd.cut(pd.to_numeric(d[y], errors="coerce"), bins=bins, duplicates="drop")
        pv = d.pivot_table(index=y, columns=x, values="ID" if "ID" in d.columns else d.columns[0], aggfunc="count", fill_value=0)
        if pv.empty:
            st.info("No data for this combination.")
        else:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(pv, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            ax.set_xlabel(x); ax.set_ylabel(y); st.pyplot(fig)
    st.caption("Heatmaps trade detail for clarity — great for **counts**, but they depend on **binning choices**.")

# ---------- Scatter (no binning) ----------
with tab_scatter:
    st.subheader("Fine-grained view without binning (color by outcome)")
    st.caption("Choose two continuous variables (e.g., Age and Distance to a pump). Filter using the sidebar to see patterns strengthen/weaken.")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric variables.")
    else:
        x = st.selectbox("X variable", numeric_cols, index=max(0, numeric_cols.index("Age") if "Age" in numeric_cols else 0))
        default_y = next((c for c in numeric_cols if c.startswith("Distance to Pump")), numeric_cols[0])
        y = st.selectbox("Y variable", numeric_cols, index=max(0, numeric_cols.index(default_y) if default_y in numeric_cols else 1))
        d = apply_filters(df).dropna(subset=[x,y]).copy()
        fig, ax = plt.subplots(figsize=(9,5))
        if "Health Status" in d.columns:
            for status, sub in d.groupby("Health Status"):
                ax.scatter(sub[x], sub[y], alpha=0.65, label=status)
            ax.legend(title="Health Status")
        else:
            ax.scatter(d[x], d[y], alpha=0.65)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title("Scatter without binning")
        st.pyplot(fig)

    st.markdown("""
**From visuals to hypothesis:** Based on the distributions, heatmap, and scatter, what pattern seems most likely?
""")

# ---------- Hypothesis & stats (correlation, not causation) ----------
with tab_stats:
    st.subheader("Confirming a *statistical correlation* (not causation)")
    st.markdown("""
Two reminders:
1) We can test for **association/correlation**, but that’s not proof of **causation** (e.g., a superspreading event near one pump could mimic a distance effect).  
2) A simple **model/test** helps check whether the pattern is strong enough to stand out from noise.
""")

    from scipy import stats

    # A) Chi-squared: Health Status vs Nearest Pump
    if "Health Status" in df.columns and "Nearest Pump" in df.columns:
        st.markdown("**A. Chi-squared: Is Health Status associated with Nearest Pump?**")
        ctab = pd.crosstab(df["Health Status"], df["Nearest Pump"])
        chi2, p, dof, exp = stats.chi2_contingency(ctab)
        st.write(ctab)
        c1, c2, c3 = st.columns(3)
        c1.metric("Chi-squared", f"{chi2:.2f}"); c2.metric("df", f"{dof}"); c3.metric("p-value", f"{p:.3g}")
        st.caption("A small p-value suggests an association (in this sample). Consider confounding and sampling variation.")

    st.divider()

    # B) t-test: compare mean distances between two pumps
    st.markdown("**B. t-test: Compare mean distances between two pumps**")
    p1 = st.selectbox("Pump 1", ["Pump A","Pump B","Pump C","Pump D"], key="p1_stats", index=0)
    p2 = st.selectbox("Pump 2", ["Pump A","Pump B","Pump C","Pump D"], key="p2_stats", index=1)
    if p1 != p2:
        c1 = f"Distance to {p1}"; c2 = f"Distance to {p2}"
        if c1 in df.columns and c2 in df.columns:
            x = pd.to_numeric(df[c1], errors="coerce").dropna()
            y = pd.to_numeric(df[c2], errors="coerce").dropna()
            if len(x) > 2 and len(y) > 2:
                t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                col1, col2 = st.columns(2); col1.metric("t-statistic", f"{t:.2f}"); col2.metric("p-value", f"{p:.3g}")
        else:
            st.info("Distance columns not available.")

    st.divider()

    # C) Simple logistic regression with interaction
    st.markdown("**C. Logistic regression with an interaction (illustration)**")
    st.caption("Outcome: Severe vs others. Predictors: Age, Distance to chosen pump, and Age×Distance.")
    if "Health Status" in df.columns and "Age" in df.columns:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        pump_for_model = st.selectbox("Use distance to:", ["Pump A","Pump B","Pump C","Pump D"], index=0)
        dist_col = f"Distance to {pump_for_model}"
        if dist_col in df.columns:
            y = df["Health Status"].astype(str).str.lower().str.contains("severe").astype(int)
            X = pd.DataFrame({
                "Age": pd.to_numeric(df["Age"], errors="coerce"),
                "Dist": pd.to_numeric(df[dist_col], errors="coerce"),
            })
            X["Age_x_Dist"] = X["Age"] * X["Dist"]
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]; y2 = y[mask]
            if len(X) >= 20 and y2.nunique() == 2:
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
                model.fit(X, y2)
                coef = model.named_steps["logisticregression"].coef_[0]
                names = ["Age (std)", "Distance (std)", "Age×Distance (std)"]
                odds = np.exp(coef)
                st.write(pd.DataFrame({"Feature": names, "Log-odds": coef, "Odds ratio (≈)": odds}))
                st.markdown("_Interaction > 0_: distance effect grows with age (or vice versa). Interpret cautiously.")
            else:
                st.info("Not enough complete rows to fit the model.")
        else:
            st.info("Distance column not found for the selected pump.")

st.divider()
st.caption(f"Dataset: {data_path} • Synthetic teaching dataset. Exploratory first; models test correlation, not causation.")

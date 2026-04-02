import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.guided_mode import (
    GUIDED_STEPS,
    backfill_session_state_defaults,
    build_default_session_state,
    count_completed_steps,
    get_current_guided_step,
    get_next_step_id,
    get_previous_step_id,
    get_step_by_id,
    is_step_complete,
    update_guided_answer,
)
from src.reporting import (
    MAX_REPORT_PLOTS,
    build_report_plot_entry,
    build_report_plot_id,
    remove_selected_plot,
    replace_selected_plot,
    upsert_selected_plot,
)
from src.i18n import translate
from src.data_schema import (
    DATA_DIR,
    get_display_column_label,
    get_display_label,
    localize_category_labels,
    localize_column_labels,
    localize_dataframe_columns,
    normalize_dataframe,
)


DISPLAY_VALUE_COLUMNS = (
    "Gender",
    "Occupation",
    "Raw Vegetable Consumption",
    "Nearest Pump",
    "Health Status",
    "Household Size Category",
)


def initialize_session_state(session_state=None):
    state = st.session_state if session_state is None else session_state
    backfill_session_state_defaults(state, build_default_session_state())


def load_data():
    language = st.session_state["language"]
    for path in [
        DATA_DIR / "cholera_dataset.csv",
        DATA_DIR / "cholera_datensatz_de.csv",
        DATA_DIR / "cholera_dataset_en.csv",
    ]:
        if not path.exists():
            continue
        try:
            return pd.read_csv(path), str(path.relative_to(DATA_DIR.parent))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError, ValueError) as exc:
            st.error(f"{translate('data.load_error', language)} ({path.name}: {exc})")
            st.stop()
    st.error(translate("data.load_error", language))
    st.stop()


def categorize_household_size(_df):
    if "Household Size" not in _df.columns:
        return _df
    _df = _df.copy()
    conds = [
        (_df["Household Size"] <= 2),
        (_df["Household Size"] <= 4),
        (_df["Household Size"] <= 6),
        (_df["Household Size"] > 6),
    ]
    _df["Household Size Category"] = np.select(conds, ["1-2", "3-4", "5-6", "7+"], default="Unknown")
    return _df


def build_app_language():
    initialize_session_state()
    return st.session_state["language"]


def localize_value(category, value, language):
    if pd.isna(value):
        return value
    return get_display_label(category, value, language)


def localize_dataframe(df, language):
    localized = df.copy()
    for column in DISPLAY_VALUE_COLUMNS:
        if column in localized.columns:
            localized[column] = localized[column].map(lambda value: localize_value(column, value, language))
    return localized


def format_column_option(value, language):
    return get_display_column_label(value, language)


def format_axis_label(value, language):
    return get_display_column_label(value, language)


def rerun_app():
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def set_guided_step(step_id):
    st.session_state["guided_step"] = step_id


def sync_guided_answer(field_id, widget_key):
    update_guided_answer(st.session_state, field_id, st.session_state[widget_key])


def set_report_selected_plots(selected_plots):
    st.session_state["selected_plots"] = selected_plots


def set_report_error(message):
    st.session_state["report_ready_state"]["last_error"] = message


def clear_report_error():
    set_report_error(None)


def save_report_plot(plot_entry):
    try:
        set_report_selected_plots(upsert_selected_plot(st.session_state["selected_plots"], plot_entry))
        clear_report_error()
        rerun_app()
    except ValueError as exc:
        set_report_error(str(exc))


def remove_report_plot(plot_id):
    set_report_selected_plots(remove_selected_plot(st.session_state["selected_plots"], plot_id))
    clear_report_error()
    rerun_app()


def replace_report_plot(plot_id, plot_entry):
    try:
        set_report_selected_plots(replace_selected_plot(st.session_state["selected_plots"], plot_id, plot_entry))
        clear_report_error()
        rerun_app()
    except ValueError as exc:
        set_report_error(str(exc))


def render_report_sidebar(app_language):
    selected_plots = st.session_state["selected_plots"]
    st.sidebar.subheader(translate("report.sidebar.title", app_language))
    st.sidebar.caption(
        translate("report.sidebar.selected_count", app_language).format(
            count=len(selected_plots),
            max_plots=MAX_REPORT_PLOTS,
        )
    )
    st.sidebar.caption(translate("report.sidebar.hint", app_language))

    report_error = st.session_state["report_ready_state"].get("last_error")
    if report_error:
        st.sidebar.error(translate("report.error.prefix", app_language).format(message=report_error))

    if not selected_plots:
        st.sidebar.info(translate("report.sidebar.empty", app_language))
        return

    for plot in selected_plots:
        st.sidebar.markdown(f"**{plot['title']}**")
        st.sidebar.caption(f"{plot['plot_type']} • {plot['caption']}")
        st.sidebar.caption(
            ", ".join(f"{key}: {value}" for key, value in plot["parameters"].items())
            or translate("report.sidebar.no_parameters", app_language)
        )
        st.sidebar.button(
            translate("report.sidebar.remove", app_language),
            key=f"report_remove_{plot['plot_id']}",
            on_click=remove_report_plot,
            args=(plot["plot_id"],),
        )
        st.sidebar.divider()


def render_report_plot_controls(app_language, plot_type, title, caption, parameters, fig):
    plot_id = build_report_plot_id(plot_type, parameters)
    selected_plots = st.session_state["selected_plots"]
    selected_plot_ids = [plot["plot_id"] for plot in selected_plots]

    def current_plot_entry():
        return build_report_plot_entry(
            plot_id=plot_id,
            plot_type=plot_type,
            title=title,
            caption=caption,
            parameters=parameters,
            fig=fig,
        )

    if plot_id in selected_plot_ids:
        if st.button(
            translate("report.controls.update", app_language),
            key=f"report_update_{plot_id}",
        ):
            save_report_plot(current_plot_entry())
        return

    if len(selected_plots) < MAX_REPORT_PLOTS:
        if st.button(
            translate("report.controls.add", app_language),
            key=f"report_add_{plot_id}",
        ):
            save_report_plot(current_plot_entry())
        return

    replacement_options = selected_plot_ids
    replacement_labels = {
        plot["plot_id"]: f"{plot['title']} ({plot['plot_type']})" for plot in selected_plots
    }
    replacement_target = st.selectbox(
        translate("report.controls.replace_label", app_language),
        replacement_options,
        key=f"report_replace_target_{plot_id}",
        format_func=lambda value: replacement_labels.get(value, value),
    )
    if st.button(
        translate("report.controls.replace_button", app_language),
        key=f"report_replace_{plot_id}",
    ):
        replace_report_plot(replacement_target, current_plot_entry())


def build_selected_plot_parameters(plot_type, **parameters):
    filters = parameters.get("filters", {})
    other_parameters = {key: value for key, value in parameters.items() if key != "filters"}
    return {
        "plot_type": plot_type,
        "filters": filters,
        **other_parameters,
    }


def render_guided_sidebar(app_language):
    st.sidebar.subheader(translate("guided.sidebar.title", app_language))
    st.sidebar.checkbox(translate("guided.sidebar.enabled", app_language), key="guided_mode_enabled")

    if not st.session_state["guided_mode_enabled"]:
        st.sidebar.caption(translate("guided.sidebar.disabled", app_language))
        st.sidebar.markdown(translate("guided.sidebar.empty_state", app_language))
        return

    st.sidebar.text_input(
        translate("guided.sidebar.student_name", app_language),
        key="student_name",
    )
    st.sidebar.text_input(
        translate("guided.sidebar.group_name", app_language),
        key="group_name",
    )

    guided_step_id = get_current_guided_step(st.session_state)
    guided_step = get_step_by_id(guided_step_id) or GUIDED_STEPS[0]
    guided_answers = st.session_state["guided_answers"]
    completed_steps = count_completed_steps(guided_answers)
    total_steps = len(GUIDED_STEPS)
    step_title = translate(guided_step["title_msg_id"], app_language)

    st.sidebar.caption(
        translate("guided.sidebar.current_step", app_language).format(step=step_title)
    )
    st.sidebar.caption(
        translate("guided.sidebar.completed_steps", app_language).format(
            completed=completed_steps,
            total=total_steps,
        )
    )
    st.sidebar.caption(translate("guided.sidebar.progress", app_language))
    st.sidebar.progress(completed_steps / total_steps if total_steps else 0.0)

    if is_step_complete(guided_step, guided_answers):
        st.sidebar.success(translate("guided.sidebar.complete", app_language))
    else:
        st.sidebar.info(translate("guided.sidebar.incomplete", app_language))

    st.sidebar.markdown(f"**{step_title}**")
    st.sidebar.caption(translate(guided_step["prompt_msg_id"], app_language))

    for field in guided_step["fields"]:
        widget_key = f"guided_{field['field_id']}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = guided_answers.get(field["field_id"], "")
        st.sidebar.text_area(
            translate(field["label_msg_id"], app_language),
            key=widget_key,
            on_change=sync_guided_answer,
            args=(field["field_id"], widget_key),
            height=120,
        )

    previous_step_id = get_previous_step_id(guided_step_id)
    next_step_id = get_next_step_id(guided_step_id)
    previous_disabled = previous_step_id is None
    next_disabled = next_step_id is None
    if st.sidebar.button(
        translate("guided.sidebar.previous", app_language),
        key="guided_previous_step",
        disabled=previous_disabled,
    ):
        set_guided_step(previous_step_id)
        rerun_app()
    if st.sidebar.button(
        translate("guided.sidebar.next", app_language),
        key="guided_next_step",
        disabled=next_disabled,
    ):
        set_guided_step(next_step_id)
        rerun_app()


def display_option(category, value, language):
    if value == "All":
        return translate("common.all", language)
    return get_display_label(category, value, language)


def main():
    app_language = build_app_language()
    st.set_page_config(page_title=translate("app.title", app_language), layout="wide")

    st.title(f"🦠 {translate('app.title', app_language)}")
    st.sidebar.selectbox(
        translate("sidebar.language", app_language),
        ["de", "en"],
        key="language",
        format_func=lambda value: translate(f"language.{value}", app_language),
    )
    app_language = st.session_state["language"]

    render_guided_sidebar(app_language)
    render_report_sidebar(app_language)
    render_report_sidebar()

    st.markdown(translate("app.intro", app_language))
    st.info(translate("app.focus", app_language))

    # ---------- Data ----------
    df, data_path = load_data()
    df = normalize_dataframe(df)
    df = categorize_household_size(df)

    # ---------- Sidebar Controls ----------
    st.sidebar.header(translate("sidebar.filters", app_language))
    age_min = int(np.nanmin(df["Age"])) if "Age" in df.columns else 0
    age_max = int(np.nanmax(df["Age"])) if "Age" in df.columns else 80
    age_range = st.sidebar.slider(translate("sidebar.age_range", app_language), 0, max(80, age_max), (age_min, age_max), 1)

    gender_options = ["All"] + (sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else [])
    occupation_options = ["All"] + (sorted(df["Occupation"].dropna().unique().tolist()) if "Occupation" in df.columns else [])
    household_options = ["All"] + (["1-2", "3-4", "5-6", "7+"] if "Household Size Category" in df.columns else [])
    raw_veg_options = ["All"] + (sorted(df["Raw Vegetable Consumption"].dropna().unique().tolist()) if "Raw Vegetable Consumption" in df.columns else [])
    nearest_pump_options = ["All", "Pump A", "Pump B", "Pump C", "Pump D"]

    gender = st.sidebar.selectbox(
        translate("sidebar.gender", app_language),
        gender_options,
        index=0,
        format_func=lambda value: display_option("Gender", value, app_language),
    )
    occupation = st.sidebar.selectbox(
        translate("sidebar.occupation", app_language),
        occupation_options,
        index=0,
        format_func=lambda value: display_option("Occupation", value, app_language),
    )
    hh_cat = st.sidebar.selectbox(
        translate("sidebar.household_size", app_language),
        household_options,
        index=0,
        format_func=lambda value: display_option("Household Size Category", value, app_language),
    )
    rv_col = "Raw Vegetable Consumption" if "Raw Vegetable Consumption" in df.columns else None
    raw_veg = (
        st.sidebar.selectbox(
            translate("sidebar.raw_vegetables", app_language),
            raw_veg_options,
            index=0,
            format_func=lambda value: display_option("Raw Vegetable Consumption", value, app_language),
        )
        if rv_col
        else "All"
    )
    nearest_pump = st.sidebar.selectbox(
        translate("sidebar.nearest_pump", app_language),
        nearest_pump_options,
        index=0,
        format_func=lambda value: display_option("Nearest Pump", value, app_language),
    )
    bins = st.sidebar.slider(translate("sidebar.bin_width", app_language), 2, 20, 6, 1)

    def apply_filters(_df):
        d = _df.copy()
        if "Age" in d.columns:
            d = d[(d["Age"] >= age_range[0]) & (d["Age"] <= age_range[1])]
        if gender != "All" and "Gender" in d.columns:
            d = d[d["Gender"] == gender]
        if occupation != "All" and "Occupation" in d.columns:
            d = d[d["Occupation"] == occupation]
        if hh_cat != "All" and "Household Size Category" in d.columns:
            d = d[d["Household Size Category"] == hh_cat]
        if raw_veg != "All" and rv_col:
            d = d[d[rv_col] == raw_veg]
        if nearest_pump != "All" and "Nearest Pump" in d.columns:
            d = d[d["Nearest Pump"] == nearest_pump]
        return d

    # ---------- Tabs ----------
    tab_intro, tab_dist, tab_heat, tab_scatter, tab_stats = st.tabs(
        [
            translate("tab.overview", app_language),
            translate("tab.distributions", app_language),
            translate("tab.heatmap", app_language),
            translate("tab.scatter", app_language),
            translate("tab.stats", app_language),
        ]
    )

    # ---------- Overview ----------
    with tab_intro:
        st.subheader(translate("overview.subtitle", app_language))
        filtered_df = apply_filters(df)
        filtered_display_df = localize_dataframe_columns(localize_dataframe(filtered_df.head(20), app_language), app_language)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(filtered_display_df)
        with c2:
            st.metric(translate("common.filtered_rows", app_language), len(filtered_df))
            st.metric(translate("common.total_rows", app_language), len(df))
            st.metric(translate("common.columns", app_language), len(df.columns))
            if "Health Status" in filtered_df.columns:
                st.write(translate("overview.health_outcome", app_language))
                st.write(localize_dataframe(filtered_df[["Health Status"]], app_language)["Health Status"].value_counts(dropna=False))
        st.markdown(translate("overview.missing_note", app_language))

        miss = filtered_df.isna().sum().sort_values(ascending=False)
        if miss.sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 3))
            miss_display = miss.copy()
            miss_display.index = localize_column_labels(miss_display.index, app_language)
            ax.bar(miss_display.index, miss_display.values)
            ax.set_xticklabels(miss_display.index, rotation=45, ha="right")
            ax.set_ylabel(translate("overview.missing_ylabel", app_language))
            st.pyplot(fig)

    # ---------- Distributions (stacked histogram / bar) ----------
    with tab_dist:
        st.subheader(translate("distributions.subtitle", app_language))
        st.caption(translate("distributions.caption", app_language))
        exclude = {"ID", "Home Location X", "Home Location Y"}
        candidates = [c for c in df.columns if c not in exclude]
        var = st.selectbox(
            translate("common.variable", app_language),
            candidates,
            index=(candidates.index("Age") if "Age" in candidates else 0),
            format_func=lambda value: format_column_option(value, app_language),
        )

        d = apply_filters(df).copy()
        if var not in d.columns or "Health Status" not in d.columns:
            st.info(translate("distributions.need_variable", app_language))
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            if pd.api.types.is_numeric_dtype(d[var]):
                groups = [g[var].dropna().values for _, g in d.groupby("Health Status")]
                labels = [get_display_label("Health Status", s, app_language) for s, _ in d.groupby("Health Status")]
                ax.hist(groups, bins=bins, stacked=True, label=labels)
                ax.set_xlabel(format_axis_label(var, app_language))
                ax.set_ylabel(translate("common.count", app_language))
            else:
                grouped = d.groupby([var, "Health Status"]).size().unstack(fill_value=0)
                grouped.index = localize_category_labels(var, grouped.index, app_language)
                grouped.columns = localize_category_labels("Health Status", grouped.columns, app_language)
                grouped.plot(kind="bar", stacked=True, ax=ax)
                ax.set_xlabel(format_axis_label(var, app_language))
                ax.set_ylabel(translate("common.count", app_language))
            ax.set_title(translate("distributions.title", app_language))
            ax.legend(title=translate("common.health_status", app_language))
            st.pyplot(fig)
            distribution_parameters = build_selected_plot_parameters(
                "distribution",
                variable=var,
                bins=bins,
                filters={
                    "age_range": list(age_range),
                    "gender": gender,
                    "occupation": occupation,
                    "household_size": hh_cat,
                    "raw_veg": raw_veg,
                    "nearest_pump": nearest_pump,
                },
            )
            render_report_plot_controls(
                app_language,
                "distribution",
                translate("distributions.title", app_language),
                translate("distributions.caption", app_language),
                distribution_parameters,
                fig,
            )
            plt.close(fig)

        st.markdown(translate("distributions.discuss", app_language))

    # ---------- Heatmap (compare two variables, with binning for continuous) ----------
    with tab_heat:
        st.subheader(translate("heatmap.subtitle", app_language))
        options = [c for c in df.columns if c not in exclude and c != "Health Status"]
        if len(options) < 2:
            st.info(translate("heatmap.need_two", app_language))
        else:
            x = st.selectbox(
                translate("common.x", app_language),
                options,
                index=0,
                format_func=lambda value: format_column_option(value, app_language),
            )
            y = st.selectbox(
                translate("common.y", app_language),
                options,
                index=1,
                format_func=lambda value: format_column_option(value, app_language),
            )
            d = apply_filters(df).dropna(subset=[x, y]).copy()
            if pd.api.types.is_numeric_dtype(d[x]):
                d[x] = pd.cut(pd.to_numeric(d[x], errors="coerce"), bins=bins, duplicates="drop")
            if pd.api.types.is_numeric_dtype(d[y]):
                d[y] = pd.cut(pd.to_numeric(d[y], errors="coerce"), bins=bins, duplicates="drop")
            pv = d.pivot_table(index=y, columns=x, values="ID" if "ID" in d.columns else d.columns[0], aggfunc="count", fill_value=0)
            if pv.empty:
                st.info(translate("heatmap.no_data", app_language))
            else:
                pv_display = pv.copy()
                pv_display.index = localize_category_labels(y, pv_display.index, app_language) if not pd.api.types.is_numeric_dtype(d[y]) else pv_display.index
                pv_display.columns = localize_category_labels(x, pv_display.columns, app_language) if not pd.api.types.is_numeric_dtype(d[x]) else pv_display.columns
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(pv_display, annot=True, fmt="d", cmap="coolwarm", ax=ax)
                ax.set_xlabel(format_axis_label(x, app_language))
                ax.set_ylabel(format_axis_label(y, app_language))
                st.pyplot(fig)
                heatmap_parameters = build_selected_plot_parameters(
                    "heatmap",
                    x=x,
                    y=y,
                    bins=bins,
                    filters={
                        "age_range": list(age_range),
                        "gender": gender,
                        "occupation": occupation,
                        "household_size": hh_cat,
                        "raw_veg": raw_veg,
                        "nearest_pump": nearest_pump,
                    },
                )
                render_report_plot_controls(
                    app_language,
                    "heatmap",
                    translate("heatmap.subtitle", app_language),
                    translate("heatmap.caption", app_language),
                    heatmap_parameters,
                    fig,
                )
                plt.close(fig)
        st.caption(translate("heatmap.caption", app_language))

    # ---------- Scatter (no binning) ----------
    with tab_scatter:
        st.subheader(translate("scatter.subtitle", app_language))
        st.caption(translate("scatter.caption", app_language))
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            st.info(translate("scatter.need_two", app_language))
        else:
            x = st.selectbox(
                translate("common.x_variable", app_language),
                numeric_cols,
                index=max(0, numeric_cols.index("Age") if "Age" in numeric_cols else 0),
                format_func=lambda value: format_column_option(value, app_language),
            )
            default_y = next((c for c in numeric_cols if c.startswith("Distance to Pump")), numeric_cols[0])
            y = st.selectbox(
                translate("common.y_variable", app_language),
                numeric_cols,
                index=max(0, numeric_cols.index(default_y) if default_y in numeric_cols else 1),
                format_func=lambda value: format_column_option(value, app_language),
            )
            d = apply_filters(df).dropna(subset=[x, y]).copy()
            fig, ax = plt.subplots(figsize=(9, 5))
            if "Health Status" in d.columns:
                for status, sub in d.groupby("Health Status"):
                    ax.scatter(sub[x], sub[y], alpha=0.65, label=get_display_label("Health Status", status, app_language))
                ax.legend(title=translate("common.health_status", app_language))
            else:
                ax.scatter(d[x], d[y], alpha=0.65)
            ax.set_xlabel(format_axis_label(x, app_language))
            ax.set_ylabel(format_axis_label(y, app_language))
            ax.set_title(translate("scatter.title", app_language))
            st.pyplot(fig)
            scatter_parameters = build_selected_plot_parameters(
                "scatter",
                x=x,
                y=y,
                filters={
                    "age_range": list(age_range),
                    "gender": gender,
                    "occupation": occupation,
                    "household_size": hh_cat,
                    "raw_veg": raw_veg,
                    "nearest_pump": nearest_pump,
                },
            )
            render_report_plot_controls(
                app_language,
                "scatter",
                translate("scatter.subtitle", app_language),
                translate("scatter.caption", app_language),
                scatter_parameters,
                fig,
            )
            plt.close(fig)

        st.markdown(translate("scatter.discuss", app_language))

    # ---------- Hypothesis & stats (correlation, not causation) ----------
    with tab_stats:
        st.subheader(translate("stats.subtitle", app_language))
        st.markdown(translate("stats.reminders", app_language))

        from scipy import stats

        d = apply_filters(df).copy()

        if d.empty:
            st.info(translate("stats.no_rows", app_language))
        else:
            st.caption(translate("stats.filtered_subset", app_language).format(rows=len(d)))

        # A) Chi-squared: Health Status vs Nearest Pump
        if "Health Status" in d.columns and "Nearest Pump" in d.columns:
            st.markdown(translate("stats.chi_squared", app_language))
            ctab = pd.crosstab(d["Health Status"], d["Nearest Pump"])
            if ctab.shape[0] >= 2 and ctab.shape[1] >= 2:
                chi2, p, dof, exp = stats.chi2_contingency(ctab)
                ctab_display = ctab.copy()
                ctab_display.index = [get_display_label("Health Status", value, app_language) for value in ctab_display.index]
                ctab_display.columns = [get_display_label("Nearest Pump", value, app_language) for value in ctab_display.columns]
                st.write(ctab_display)
                c1, c2, c3 = st.columns(3)
                c1.metric(translate("stats.metric.chi_squared", app_language), f"{chi2:.2f}")
                c2.metric(translate("stats.metric.df", app_language), f"{dof}")
                c3.metric(translate("stats.metric.p_value", app_language), f"{p:.3g}")
                st.caption(translate("stats.chi_squared_question", app_language))
            else:
                st.info(translate("stats.need_two_groups", app_language))

        st.divider()

        # B) Simple logistic regression with interaction
        st.markdown(translate("stats.logistic", app_language))
        st.caption(translate("stats.logistic_caption", app_language))
        if "Health Status" in d.columns and "Age" in d.columns:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            pump_for_model = st.selectbox(
                translate("stats.use_distance", app_language),
                ["Pump A", "Pump B", "Pump C", "Pump D"],
                index=0,
                format_func=lambda value: display_option("Nearest Pump", value, app_language),
            )
            dist_col = f"Distance to {pump_for_model}"
            if dist_col in d.columns:
                severe_statuses = {"Severe Illness", "Death"}
                y = d["Health Status"].isin(severe_statuses).astype(int)
                X = pd.DataFrame({
                    "Age": pd.to_numeric(d["Age"], errors="coerce"),
                    "Dist": pd.to_numeric(d[dist_col], errors="coerce"),
                })
                X["Age_x_Dist"] = X["Age"] * X["Dist"]
                mask = X.notna().all(axis=1) & y.notna()
                X = X[mask]
                y2 = y[mask]
                if len(X) >= 20 and y2.nunique() == 2:
                    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
                    model.fit(X, y2)
                    coef = model.named_steps["logisticregression"].coef_[0]
                    names = [
                        translate("stats.feature.age_std", app_language),
                        translate("stats.feature.distance_std", app_language),
                        translate("stats.feature.age_x_distance_std", app_language),
                    ]
                    odds = np.exp(coef)
                    st.write(
                        pd.DataFrame(
                            {
                                translate("stats.table.feature", app_language): names,
                                translate("stats.table.log_odds", app_language): coef,
                                translate("stats.table.odds_ratio", app_language): odds,
                            }
                        )
                    )
                    st.caption(translate("stats.model_question", app_language))
                    st.markdown(translate("stats.model_interpretation", app_language))
                else:
                    st.info(translate("stats.need_complete_rows", app_language))
            else:
                st.info(translate("stats.distance_missing", app_language))

    st.divider()
    st.caption(translate("footer.dataset", app_language).format(data_path=data_path))


if __name__ == "__main__":
    main()

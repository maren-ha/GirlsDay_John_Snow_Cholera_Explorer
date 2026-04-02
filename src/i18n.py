STRINGS = {
    "de": {
        "app.title": "Epidemiologischer Daten-Explorer — London 1854",
        "app.subtitle": "Eine kurze Einführung in den Ausbruch von 1854.",
        "app.intro": (
            "**Ausbruch in London (1854).**  \n"
            "Eine ansteckende Krankheit mit Symptomen wie Durchfall, Übelkeit, Erbrechen und starkem Flüssigkeitsverlust. "
            "Innerhalb weniger Tage erkranken Hunderte; viele sterben. Die Ursache ist unbekannt - viele glauben an \"schlechte Luft\". "
            "Angst, Gerüchte und Verwirrung breiten sich aus.\n\n"
            "**Deine Rolle.**  \n"
            "Du schlüpfst in die Rolle von Wissenschaftlerinnen und Wissenschaftlern und versuchst mit Daten, Modellen und Experimenten herauszufinden:  \n"
            "- Wie genau verbreitet sich die Krankheit?  \n"
            "- Wer ist besonders gefährdet?  \n"
            "- Wie können wir sie stoppen?\n\n"
            "Vier Aufgaben in der Untersuchung: **Populationsdynamik-Simulation**, **Genom-Stämme analysieren**, "
            "**Antikörper-Studien** und **Epidemiologische Datenanalyse** (diese App)."
        ),
        "app.focus": "Diese App konzentriert sich auf die Spur **Epidemiologische Datenanalyse**. Starte mit explorativen Visualisierungen, formuliere eine Hypothese und prüfe sie dann mit einfachen statistischen Tests auf Korrelation (nicht Kausalität).",
        "sidebar.language": "Sprache",
        "sidebar.filters": "Filter",
        "sidebar.age_range": "Altersbereich",
        "sidebar.gender": "Geschlecht",
        "sidebar.occupation": "Beruf",
        "sidebar.household_size": "Haushaltsgröße",
        "sidebar.raw_vegetables": "Rohkost",
        "sidebar.nearest_pump": "Nächste Pumpe",
        "sidebar.bin_width": "Bin-Breite (numerische Variablen)",
        "language.de": "Deutsch",
        "language.en": "Englisch",
        "tab.overview": "Überblick",
        "tab.distributions": "Verteilungen (gestapeltes Histogramm)",
        "tab.heatmap": "Heatmap (zwei Variablen vergleichen)",
        "tab.scatter": "Streudiagramm (ohne Binning)",
        "tab.stats": "Hypothese & Statistik",
        "overview.subtitle": "Ein kurzer Blick auf die Daten",
        "overview.filtered_rows": "Gefilterte Zeilen",
        "overview.total_rows": "Gesamtzeilen",
        "overview.columns": "Spalten",
        "overview.health_outcome": "Gesundheitsergebnis (gefilterte Häufigkeiten)",
        "overview.missing_note": "**Fehlende Werte** sind in realen Datensätzen normal. Denk bei jeder Grafik daran.",
        "overview.missing_ylabel": "Fehlend",
        "distributions.subtitle": "Verteilung möglicher Risikofaktoren - Gesundheitszustand hervorheben",
        "distributions.caption": "Wähle eine Variable. Wenn sie kontinuierlich ist, bilden wir Klassen - die **Bin-Breite** ist also wichtig. Das ist explorativ.",
        "distributions.need_variable": "Für diese Grafik werden die ausgewählte Variable und der Gesundheitszustand benötigt.",
        "distributions.title": "Gestapelte Verteilung nach Gesundheitszustand",
        "distributions.discuss": (
            "**Diskutiere:**  \n"
            "- Welche Gruppen/Bins wirken stärker betroffen?  \n"
            "- Könnte eine andere Variable dahinterstecken (Konfundierung)?  \n"
            "- Wie verändert die **Bin-Breite** deinen Eindruck bei kontinuierlichen Variablen?"
        ),
        "heatmap.subtitle": "Zwei Variablen direkt vergleichen (mit Binning für kontinuierliche Variablen)",
        "heatmap.need_two": "Zum Vergleichen werden mindestens zwei Variablen benötigt.",
        "heatmap.no_data": "Für diese Kombination liegen keine Daten vor.",
        "heatmap.caption": "Heatmaps vereinfachen zugunsten der Klarheit - gut für **Anzahlen**, aber sie hängen von der **Bin-Wahl** ab.",
        "scatter.subtitle": "Fein aufgelöster Blick ohne Binning (Farbe nach Ergebnis)",
        "scatter.caption": "Wähle zwei kontinuierliche Variablen (z. B. Alter und Entfernung zu einer Pumpe). Nutze die Seitenleiste, um zu sehen, wie sich Muster verstärken oder abschwächen.",
        "scatter.need_two": "Es werden mindestens zwei numerische Variablen benötigt.",
        "scatter.title": "Streudiagramm ohne Binning",
        "scatter.discuss": (
            "**Von den Visualisierungen zur Hypothese:** Welches Muster scheint anhand der Verteilungen, der Heatmap und des Streudiagramms am wahrscheinlichsten?"
        ),
        "stats.subtitle": "Eine *statistische Korrelation* bestätigen (nicht Kausalität)",
        "stats.reminders": (
            "Zwei Erinnerungen:\n"
            "1) Wir können **Assoziation/Korrelation** testen, aber das ist kein Beweis für **Kausalität** "
            "(z. B. könnte ein Superspreading-Ereignis in der Nähe einer Pumpe einen Entfernungseffekt imitieren).  \n"
            "2) Ein einfaches **Modell/Verfahren** hilft zu prüfen, ob das Muster stark genug gegen das Rauschen hervortritt."
        ),
        "stats.no_rows": "Keine Zeilen passen zu den aktuellen Filtern. Weite die Filter, um die statistischen Checks auszuführen.",
        "stats.filtered_subset": "Die statistischen Checks unten verwenden den aktuell gefilterten Ausschnitt mit {rows} Zeilen.",
        "stats.chi_squared": "**A. Chi-Quadrat: Ist der Gesundheitszustand mit der nächsten Pumpe assoziiert?**",
        "stats.chi_squared_question": "Frage beantwortet: Unterscheidet sich der Schweregrad der Erkrankung je nach nächster Pumpe stärker, als wir es durch Zufall erwarten würden?",
        "stats.need_two_groups": "Nach dem Filtern werden mindestens zwei Gesundheitsgruppen und zwei Pumpengruppen für den Chi-Quadrat-Test benötigt.",
        "stats.logistic": "**B. Logistische Regression mit Interaktion (Illustration)**",
        "stats.logistic_caption": "Ergebnis: schwere Krankheit oder Tod vs. mildere Verläufe. Prädiktoren: Alter, Entfernung zur gewählten Pumpe und Alter×Entfernung.",
        "stats.use_distance": "Entfernung verwenden zu:",
        "stats.need_complete_rows": "Für das Modell werden nach dem Filtern mindestens 20 vollständige Zeilen und beide Ergebnisgruppen benötigt.",
        "stats.distance_missing": "Für die gewählte Pumpe wurde keine Entfernungs-Spalte gefunden.",
        "stats.model_question": "Frage beantwortet: Hilft die Entfernung zur gewählten Pumpe immer noch, schlechtere Ergebnisse vorherzusagen, wenn das Alter berücksichtigt wird?",
        "stats.model_interpretation": "_Interaktion > 0_: Der Entfernungs-Effekt wächst mit dem Alter (oder umgekehrt). Vorsichtig interpretieren._",
        "stats.metric.chi_squared": "Chi-Quadrat",
        "stats.metric.df": "Freiheitsgrade",
        "stats.metric.p_value": "p-Wert",
        "data.load_error": "Datensatz nicht gefunden (erwartet z. B. data/cholera_dataset_en.csv).",
        "common.all": "Alle",
        "common.count": "Anzahl",
        "common.missing": "Fehlend",
        "common.health_status": "Gesundheitszustand",
        "common.filtered_rows": "Gefilterte Zeilen",
        "common.total_rows": "Gesamtzeilen",
        "common.columns": "Spalten",
        "common.variable": "Variable",
        "common.x": "X",
        "common.y": "Y",
        "common.x_variable": "X-Variable",
        "common.y_variable": "Y-Variable",
        "common.use_distance_to": "Entfernung verwenden zu:",
        "stats.table.feature": "Merkmal",
        "stats.table.log_odds": "Log-Odds",
        "stats.table.odds_ratio": "Odds Ratio (≈)",
        "stats.feature.age_std": "Alter (std)",
        "stats.feature.distance_std": "Entfernung (std)",
        "stats.feature.age_x_distance_std": "Alter×Entfernung (std)",
        "guided.sidebar.title": "Geführter Modus",
        "guided.sidebar.enabled": "Geführten Modus einschalten",
        "guided.sidebar.disabled": "Freies Erkunden beibehalten",
        "guided.sidebar.student_name": "Name der Schülerin / des Schülers",
        "guided.sidebar.group_name": "Gruppe",
        "guided.sidebar.current_step": "Aktueller Schritt: {step}",
        "guided.sidebar.progress": "Fortschritt",
        "guided.sidebar.completed_steps": "{completed}/{total} Schritte abgeschlossen",
        "guided.sidebar.previous": "Vorheriger Schritt",
        "guided.sidebar.next": "Nächster Schritt",
        "guided.sidebar.complete": "Dieser Schritt ist abgeschlossen.",
        "guided.sidebar.incomplete": "Dieser Schritt ist noch nicht abgeschlossen.",
        "guided.sidebar.empty_state": (
            "Schalte den geführten Modus ein, um Schritt-für-Schritt-Hinweise und Antwortfelder zu sehen. "
            "Die freien Analyse-Tabs bleiben trotzdem verfügbar."
        ),
        "guided.overview.title": "Schritt 1: Überblick verschaffen",
        "guided.overview.prompt": "Schreibe eine kurze Beobachtung dazu, was dir im aktuell gefilterten Datenausschnitt zuerst auffällt.",
        "guided.overview.observation_label": "Deine erste Beobachtung",
        "guided.distribution.title": "Schritt 2: Verteilungen vergleichen",
        "guided.distribution.prompt": "Wähle im Verteilungstab eine Variable und beschreibe, welche Gruppen oder Werte besonders auffallen.",
        "guided.distribution.observation_label": "Was fällt bei der Verteilung auf?",
        "guided.comparison.title": "Schritt 3: Zwei Variablen vergleichen",
        "guided.comparison.prompt": "Nutze Heatmap oder Streudiagramm und beschreibe, welche Beziehung zwischen zwei Variablen sichtbar wird.",
        "guided.comparison.observation_label": "Welche Beziehung siehst du?",
        "guided.hypothesis.title": "Schritt 4: Eine Hypothese formulieren",
        "guided.hypothesis.prompt": "Formuliere eine überprüfbare Hypothese, die du mit den Daten testen möchtest.",
        "guided.hypothesis.field_label": "Deine Hypothese",
        "guided.stats.title": "Schritt 5: Belege prüfen",
        "guided.stats.prompt": "Schau dir den Statistik-Tab an und schreibe auf, ob die Ergebnisse deine Hypothese stützen.",
        "guided.stats.interpretation_label": "Was sagen die Statistik-Ergebnisse?",
        "guided.conclusion.title": "Schritt 6: Fazit ziehen",
        "guided.conclusion.prompt": "Schreibe ein kurzes Fazit und nenne, was du sicher sagen kannst und was offen bleibt.",
        "guided.conclusion.field_label": "Dein Fazit",
        "report.sidebar.title": "Berichtsauswahl",
        "report.sidebar.selected_count": "{count}/{max_plots} Grafiken ausgewählt",
        "report.sidebar.hint": "Wähle Grafiken in den Tabs Verteilungen, Heatmap oder Streudiagramm aus.",
        "report.sidebar.empty": "Noch keine Grafiken gespeichert.",
        "report.sidebar.remove": "Entfernen",
        "report.sidebar.no_parameters": "Keine Plot-Parameter",
        "report.controls.add": "Grafik zum Bericht hinzufügen",
        "report.controls.update": "Gespeicherte Grafik aktualisieren",
        "report.controls.replace_label": "Eine gespeicherte Grafik ersetzen",
        "report.controls.replace_button": "Ausgewählte Grafik ersetzen",
        "report.error.prefix": "Auswahlfehler: {message}",
        "guided.placeholder.title": "Geführter Modus",
        "guided.placeholder.body": "In einem späteren Schritt kommt hier die geführte Analyse mit studentischen Eingaben.",
        "export.placeholder.title": "Export",
        "export.placeholder.body": "Der Berichtsexport folgt in einem späteren Schritt.",
        "footer.dataset": "Datensatz: {data_path} • Synthetischer Lehr-Datensatz. Erst explorieren; Modelle testen Korrelation, nicht Kausalität.",
    },
    "en": {
        "app.title": "Epidemiological Data Explorer — London 1854",
        "app.intro": (
            "**Outbreak in London (1854).**  \n"
            "An infectious disease with symptoms such as diarrhea, nausea, vomiting, and severe fluid loss. Within a few days, hundreds fall ill; many die. "
            "The cause is unknown - many believe in \"bad air.\" Fear, rumors, and confusion spread.\n\n"
            "**Your role.**  \n"
            "You step into the shoes of scientists and, using data, models, and experiments, try to find out:  \n"
            "- How exactly does the disease spread?  \n"
            "- Who is particularly at risk?  \n"
            "- How can we stop it?\n\n"
            "Four tasks in the investigation: **Population dynamics simulation**, **Analyze genome strains**, **Antibody studies**, and **Epidemiological data analysis** (this app)."
        ),
        "app.focus": "This app focuses on the *Epidemiological data analysis* track. Start with exploratory visuals, form a hypothesis, then use simple statistical checks to test for correlation (not causation).",
        "sidebar.language": "Language",
        "sidebar.filters": "Filters",
        "sidebar.age_range": "Age range",
        "sidebar.gender": "Gender",
        "sidebar.occupation": "Occupation",
        "sidebar.household_size": "Household size",
        "sidebar.raw_vegetables": "Raw vegetables",
        "sidebar.nearest_pump": "Nearest pump",
        "sidebar.bin_width": "Bin width (numeric variables)",
        "language.de": "German",
        "language.en": "English",
        "tab.overview": "Overview",
        "tab.distributions": "Distributions (stacked histogram)",
        "tab.heatmap": "Heatmap (compare two variables)",
        "tab.scatter": "Scatter (no binning)",
        "tab.stats": "Hypothesis & stats",
        "overview.subtitle": "A quick look at the data",
        "overview.filtered_rows": "Filtered rows",
        "overview.total_rows": "Total rows",
        "overview.columns": "Columns",
        "overview.health_outcome": "Health outcome (filtered counts)",
        "overview.missing_note": "**Missing values** are common in real datasets. Keep them in mind when interpreting any plot.",
        "overview.missing_ylabel": "Missing",
        "distributions.subtitle": "Distribution of potential risk factors - highlight health status",
        "distributions.caption": "Select a variable. If it’s continuous, we bin it - so **bin width matters**. This is exploratory.",
        "distributions.need_variable": "Need the selected variable and Health Status to draw this plot.",
        "distributions.title": "Stacked distribution by Health Status",
        "distributions.discuss": (
            "**Discuss:**  \n"
            "- Which groups/bins look more affected?  \n"
            "- Could this be driven by another variable (confounding)?  \n"
            "- How does changing the **bin width** change your impression for continuous variables?"
        ),
        "heatmap.subtitle": "Directly compare two variables (with binning for continuous)",
        "heatmap.need_two": "Need at least two variables to compare.",
        "heatmap.no_data": "No data for this combination.",
        "heatmap.caption": "Heatmaps trade detail for clarity - great for **counts**, but they depend on **binning choices**.",
        "scatter.subtitle": "Fine-grained view without binning (color by outcome)",
        "scatter.caption": "Choose two continuous variables (e.g., Age and Distance to a pump). Filter using the sidebar to see patterns strengthen/weaken.",
        "scatter.need_two": "Need at least two numeric variables.",
        "scatter.title": "Scatter without binning",
        "scatter.discuss": (
            "**From visuals to hypothesis:** Based on the distributions, heatmap, and scatter, what pattern seems most likely?"
        ),
        "stats.subtitle": "Confirming a *statistical correlation* (not causation)",
        "stats.reminders": (
            "Two reminders:\n"
            "1) We can test for **association/correlation**, but that’s not proof of **causation** (e.g., a superspreading event near one pump could mimic a distance effect).  \n"
            "2) A simple **model/test** helps check whether the pattern is strong enough to stand out from noise."
        ),
        "stats.no_rows": "No rows match the current filters. Broaden the filters to run the statistical checks.",
        "stats.filtered_subset": "Statistical checks below use the currently filtered subset of {rows} rows.",
        "stats.chi_squared": "**A. Chi-squared: Is Health Status associated with Nearest Pump?**",
        "stats.chi_squared_question": "Question answered: does illness severity vary by nearest pump more than we would expect from random variation alone?",
        "stats.need_two_groups": "Need at least two health-status groups and two pump groups after filtering to run chi-squared.",
        "stats.logistic": "**B. Logistic regression with an interaction (illustration)**",
        "stats.logistic_caption": "Outcome: severe illness or death vs milder outcomes. Predictors: Age, Distance to chosen pump, and Age×Distance.",
        "stats.use_distance": "Use distance to:",
        "stats.need_complete_rows": "Need at least 20 complete rows and both outcome groups after filtering to fit the model.",
        "stats.distance_missing": "Distance column not found for the selected pump.",
        "stats.model_question": "Question answered: does distance to the selected pump still help predict worse outcomes after taking age into account?",
        "stats.model_interpretation": "_Interaction > 0_: distance effect grows with age (or vice versa). Interpret cautiously._",
        "stats.metric.chi_squared": "Chi-squared",
        "stats.metric.df": "df",
        "stats.metric.p_value": "p-value",
        "data.load_error": "Could not find a dataset (expected e.g. data/cholera_dataset_en.csv).",
        "common.all": "All",
        "common.count": "Count",
        "common.missing": "Missing",
        "common.health_status": "Health Status",
        "common.filtered_rows": "Filtered rows",
        "common.total_rows": "Total rows",
        "common.columns": "Columns",
        "common.variable": "Variable",
        "common.x": "X",
        "common.y": "Y",
        "common.x_variable": "X variable",
        "common.y_variable": "Y variable",
        "common.use_distance_to": "Use distance to:",
        "stats.table.feature": "Feature",
        "stats.table.log_odds": "Log-odds",
        "stats.table.odds_ratio": "Odds ratio (≈)",
        "stats.feature.age_std": "Age (std)",
        "stats.feature.distance_std": "Distance (std)",
        "stats.feature.age_x_distance_std": "Age×Distance (std)",
        "guided.sidebar.title": "Guided mode",
        "guided.sidebar.enabled": "Turn on guided mode",
        "guided.sidebar.disabled": "Keep free exploration visible",
        "guided.sidebar.student_name": "Student name",
        "guided.sidebar.group_name": "Group",
        "guided.sidebar.current_step": "Current step: {step}",
        "guided.sidebar.progress": "Progress",
        "guided.sidebar.completed_steps": "{completed}/{total} steps complete",
        "guided.sidebar.previous": "Previous step",
        "guided.sidebar.next": "Next step",
        "guided.sidebar.complete": "This step is complete.",
        "guided.sidebar.incomplete": "This step is not complete yet.",
        "guided.sidebar.empty_state": (
            "Turn on guided mode to see step-by-step prompts and answer boxes. "
            "The exploration tabs stay available either way."
        ),
        "guided.overview.title": "Step 1: Get oriented",
        "guided.overview.prompt": "Write a short observation about the first thing you notice in the current filtered slice of data.",
        "guided.overview.observation_label": "Your first observation",
        "guided.distribution.title": "Step 2: Compare distributions",
        "guided.distribution.prompt": "Pick a variable in the distribution tab and describe which groups or values stand out.",
        "guided.distribution.observation_label": "What stands out in the distribution?",
        "guided.comparison.title": "Step 3: Compare two variables",
        "guided.comparison.prompt": "Use the heatmap or scatter tab and describe the relationship you can see between two variables.",
        "guided.comparison.observation_label": "What relationship do you see?",
        "guided.hypothesis.title": "Step 4: Form a testable hypothesis",
        "guided.hypothesis.prompt": "Write a hypothesis that you can test with the data.",
        "guided.hypothesis.field_label": "Your hypothesis",
        "guided.stats.title": "Step 5: Check the evidence",
        "guided.stats.prompt": "Look at the stats tab and write whether the results support your hypothesis.",
        "guided.stats.interpretation_label": "What do the statistics suggest?",
        "guided.conclusion.title": "Step 6: Draw a conclusion",
        "guided.conclusion.prompt": "Write a short conclusion and note what you can say confidently and what still remains open.",
        "guided.conclusion.field_label": "Your conclusion",
        "report.sidebar.title": "Report selection",
        "report.sidebar.selected_count": "{count}/{max_plots} plots selected",
        "report.sidebar.hint": "Pick plots from the Distribution, Heatmap, or Scatter tabs.",
        "report.sidebar.empty": "No plots saved yet.",
        "report.sidebar.remove": "Remove",
        "report.sidebar.no_parameters": "No plot parameters",
        "report.controls.add": "Add plot to report",
        "report.controls.update": "Update saved plot",
        "report.controls.replace_label": "Replace one saved plot",
        "report.controls.replace_button": "Replace selected plot",
        "report.error.prefix": "Selection error: {message}",
        "guided.placeholder.title": "Guided mode",
        "guided.placeholder.body": "The guided analysis with student responses will land in a later task.",
        "export.placeholder.title": "Export",
        "export.placeholder.body": "Report export will follow in a later task.",
        "footer.dataset": "Dataset: {data_path} • Synthetic teaching dataset. Exploratory first; models test correlation, not causation.",
    },
}


def get_default_language():
    return "de"


def translate(key, language):
    strings_for_language = STRINGS.get(language, {})
    if key in strings_for_language:
        return strings_for_language[key]

    german_strings = STRINGS.get("de", {})
    if key in german_strings:
        return german_strings[key]

    english_strings = STRINGS.get("en", {})
    return english_strings.get(key, key)

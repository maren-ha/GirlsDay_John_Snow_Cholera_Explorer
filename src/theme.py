from html import escape
import re
from textwrap import dedent


APP_COLORS = {
    "paper": "#f6f1e7",
    "surface": "#fffdf9",
    "surface_soft": "#f3eadf",
    "surface_subtle": "#eadfce",
    "ink": "#1f2328",
    "muted": "#5c5f66",
    "accent": "#8a5a2b",
    "accent_deep": "#5d3d1f",
    "accent_soft": "#d8bd95",
    "line": "#d8c8b4",
    "success": "#3f6d52",
    "warning": "#8d5f2b",
    "shadow": "rgba(38, 27, 17, 0.10)",
}

PLOT_PALETTE = [
    APP_COLORS["accent"],
    "#6b7d7a",
    APP_COLORS["success"],
    "#b57f4a",
    "#4f5d75",
]

def build_theme_css():
    return dedent(
        f"""
        <style>
          :root {{
            --app-paper: {APP_COLORS["paper"]};
            --app-surface: {APP_COLORS["surface"]};
            --app-surface-soft: {APP_COLORS["surface_soft"]};
            --app-surface-subtle: {APP_COLORS["surface_subtle"]};
            --app-ink: {APP_COLORS["ink"]};
            --app-muted: {APP_COLORS["muted"]};
            --app-accent: {APP_COLORS["accent"]};
            --app-accent-deep: {APP_COLORS["accent_deep"]};
            --app-accent-soft: {APP_COLORS["accent_soft"]};
            --app-line: {APP_COLORS["line"]};
            --app-shadow: {APP_COLORS["shadow"]};
          }}

          .stApp {{
            background:
              radial-gradient(circle at top left, rgba(138, 90, 43, 0.08), transparent 36%),
              linear-gradient(180deg, #fbf7f0 0%, var(--app-paper) 52%, #efe5d7 100%);
            color: var(--app-ink);
          }}

          .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1180px;
          }}

          h1, h2, h3, h4 {{
            color: var(--app-ink);
            letter-spacing: -0.02em;
          }}

          h1 {{
            font-family: "Georgia", "Times New Roman", serif;
            font-weight: 700;
          }}

          .app-hero,
          .app-card,
          .app-plot-header {{
            border: 1px solid var(--app-line);
            border-radius: 1.1rem;
            background: linear-gradient(180deg, rgba(255, 253, 249, 0.98), rgba(255, 249, 242, 0.94));
            box-shadow: 0 18px 42px var(--app-shadow);
          }}

          .app-hero {{
            padding: 1.3rem 1.35rem 1.15rem;
            margin-bottom: 1rem;
            border-left: 0.35rem solid var(--app-accent);
          }}

          .app-hero-eyebrow,
          .app-card-eyebrow,
          .app-plot-eyebrow {{
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            color: var(--app-muted);
            margin-bottom: 0.4rem;
          }}

          .app-hero-title {{
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 2.25rem;
            line-height: 1.08;
            font-weight: 700;
            color: var(--app-ink);
            margin: 0;
          }}

          .app-hero-subtitle {{
            font-size: 1rem;
            color: var(--app-accent-deep);
            margin-top: 0.45rem;
            font-weight: 600;
          }}

          .app-hero-focus {{
            margin-top: 0.8rem;
            font-size: 0.98rem;
            line-height: 1.55;
            color: var(--app-ink);
          }}

          .app-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.9rem;
          }}

          .app-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            background: rgba(138, 90, 43, 0.10);
            color: var(--app-accent-deep);
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid rgba(138, 90, 43, 0.18);
          }}

          .app-card,
          .app-plot-header {{
            padding: 0.95rem 1rem;
          }}

          .app-card--compact {{
            background: linear-gradient(180deg, rgba(243, 234, 223, 0.84), rgba(255, 253, 249, 0.96));
          }}

          .app-card-title,
          .app-plot-title {{
            margin: 0;
            font-size: 1.05rem;
            line-height: 1.25;
            font-weight: 700;
            color: var(--app-ink);
          }}

          .app-card-body,
          .app-plot-caption {{
            margin-top: 0.45rem;
            font-size: 0.93rem;
            line-height: 1.55;
            color: var(--app-muted);
          }}

          .app-card-note {{
            margin-top: 0.6rem;
            color: var(--app-accent-deep);
            font-size: 0.86rem;
            font-weight: 600;
          }}

          .app-card--accent {{
            border-left: 0.28rem solid var(--app-accent);
          }}

          .app-plot-header {{
            margin-bottom: 0.65rem;
          }}

          .app-plot-title {{
            font-size: 1.02rem;
          }}

          .app-plot-caption {{
            margin-top: 0.35rem;
          }}

          .app-sidebar-note {{
            margin-top: 0.45rem;
            color: var(--app-accent-deep);
            font-size: 0.88rem;
          }}

          @media (max-width: 900px) {{
            .app-hero-title {{
              font-size: 1.8rem;
            }}

            .block-container {{
              padding-top: 1rem;
              padding-bottom: 1.4rem;
            }}
          }}
        </style>
        """
    ).strip()


def apply_app_theme():
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette=PLOT_PALETTE,
        rc={
            "figure.facecolor": APP_COLORS["paper"],
            "axes.facecolor": APP_COLORS["surface"],
            "axes.edgecolor": APP_COLORS["line"],
            "axes.labelcolor": APP_COLORS["ink"],
            "axes.titlecolor": APP_COLORS["ink"],
            "text.color": APP_COLORS["ink"],
            "xtick.color": APP_COLORS["muted"],
            "ytick.color": APP_COLORS["muted"],
            "grid.color": APP_COLORS["line"],
            "grid.alpha": 0.35,
            "legend.facecolor": APP_COLORS["surface"],
            "legend.edgecolor": APP_COLORS["line"],
        },
    )


def _format_body(body):
    if body is None:
        return ""

    escaped = escape(str(body))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", escaped)
    return escaped.replace("\n", "<br>")


def build_hero_html(title, subtitle, focus, eyebrow=None, chips=None):
    eyebrow_html = (
        f"<div class='app-hero-eyebrow'>{escape(str(eyebrow))}</div>" if eyebrow else ""
    )
    chip_html = ""
    if chips:
        chip_html = "<div class='app-chip-row'>" + "".join(
            f"<span class='app-chip'>{escape(str(chip))}</span>" for chip in chips
        ) + "</div>"

    return dedent(
        f"""
        <div class="app-hero">
          {eyebrow_html}
          <h1 class="app-hero-title">{escape(str(title))}</h1>
          <div class="app-hero-subtitle">{escape(str(subtitle))}</div>
          <div class="app-hero-focus">{_format_body(focus)}</div>
          {chip_html}
        </div>
        """
    ).strip()


def build_sidebar_card_html(title, body, eyebrow=None, note=None, accent=False):
    classes = ["app-card", "app-card--compact"]
    if accent:
        classes.append("app-card--accent")

    eyebrow_html = f"<div class='app-card-eyebrow'>{escape(str(eyebrow))}</div>" if eyebrow else ""
    note_html = f"<div class='app-card-note'>{_format_body(note)}</div>" if note else ""
    return dedent(
        f"""
        <div class="{' '.join(classes)}">
          {eyebrow_html}
          <div class="app-card-title">{escape(str(title))}</div>
          <div class="app-card-body">{_format_body(body)}</div>
          {note_html}
        </div>
        """
    ).strip()


def render_plot_header_html(title, caption, eyebrow=None, note=None):
    eyebrow_html = f"<div class='app-plot-eyebrow'>{escape(str(eyebrow))}</div>" if eyebrow else ""
    note_html = f"<div class='app-card-note'>{_format_body(note)}</div>" if note else ""
    return dedent(
        f"""
        <div class="app-plot-header">
          {eyebrow_html}
          <div class="app-plot-title">{escape(str(title))}</div>
          <div class="app-plot-caption">{_format_body(caption)}</div>
          {note_html}
        </div>
        """
    ).strip()


def style_axes(ax, grid=True):
    if ax is None:
        return None

    fig = ax.figure
    if fig is not None:
        fig.patch.set_facecolor(APP_COLORS["paper"])

    ax.set_facecolor(APP_COLORS["surface"])
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, axis="y", alpha=0.28)
    else:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(APP_COLORS["line"])
    ax.spines["bottom"].set_color(APP_COLORS["line"])
    ax.tick_params(colors=APP_COLORS["muted"])
    return ax

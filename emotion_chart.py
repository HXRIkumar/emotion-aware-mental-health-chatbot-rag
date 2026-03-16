"""
Emotion Chart Generator
Produces publication-quality matplotlib figures for Gradio gr.Plot.
Four charts in one dashboard:
  1. Emotion confidence line chart over turns
  2. Emotion category timeline (scatter)
  3. Emotion frequency bar chart
  4. Mood stability gauge
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for Colab/Gradio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Dict

# ── Emotion colour palette ────────────────────────────────────────────────
EMOTION_COLORS = {
    "sad":        "#5b9bd5",   # steel blue
    "anxious":    "#a167e8",   # purple
    "stressed":   "#e85d4a",   # coral red
    "lonely":     "#4db8c4",   # teal
    "angry":      "#f0803c",   # orange
    "overwhelmed":"#c94f7c",   # pink
    "hopeful":    "#5cb85c",   # green
    "neutral":    "#9e9e9e",   # grey
    "happy":      "#f5c542",   # yellow
}

EMOTION_ORDER = [
    "happy", "hopeful", "neutral",
    "lonely", "anxious", "overwhelmed",
    "stressed", "angry", "sad"
]

VALENCE = {
    "happy": 4, "hopeful": 3, "neutral": 2,
    "lonely": 1, "anxious": 1, "overwhelmed": 0,
    "stressed": 0, "angry": -1, "sad": -1,
}


def _apply_dark_style():
    """Apply a clean dark theme to all charts."""
    plt.rcParams.update({
        "figure.facecolor":  "#1e1e2e",
        "axes.facecolor":    "#2a2a3e",
        "axes.edgecolor":    "#44445a",
        "axes.labelcolor":   "#c8c8e8",
        "axes.titlecolor":   "#e8e8ff",
        "xtick.color":       "#a0a0c0",
        "ytick.color":       "#a0a0c0",
        "grid.color":        "#3a3a5a",
        "grid.alpha":        0.6,
        "text.color":        "#e8e8ff",
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def build_emotion_dashboard(emotion_history: List[Dict]) -> plt.Figure:
    """
    Build a 4-panel emotion analytics dashboard.

    Args:
        emotion_history: list of dicts with keys: turn, emotion, confidence, valence
                         (from EmotionTrendTracker.emotion_history)

    Returns:
        matplotlib Figure — pass directly to gr.Plot(value=fig)
    """
    _apply_dark_style()

    if not emotion_history:
        # Return a placeholder figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No conversation data yet.\nStart chatting to see your emotional trend!",
                ha="center", va="center", fontsize=14,
                color="#7f8cff", transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle("💙 Emotional Analytics Dashboard", fontsize=16, color="#e8e8ff", y=0.98)
        fig.tight_layout()
        return fig

    turns       = [e["turn"] for e in emotion_history]
    emotions    = [e["emotion"] for e in emotion_history]
    confidences = [e["confidence"] for e in emotion_history]

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1e1e2e")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Confidence line chart ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_confidence_line(ax1, turns, emotions, confidences)

    # ── Panel 2: Emotion timeline (scatter) ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_emotion_timeline(ax2, turns, emotions, confidences)

    # ── Panel 3: Frequency bar chart ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _draw_frequency_bars(ax3, emotions)

    # ── Panel 4: Mood stability / valence line ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _draw_valence_trend(ax4, turns, emotions)

    fig.suptitle("Emotional Analytics Dashboard",
                 fontsize=17, color="#e8e8ff",
                 fontweight="bold", y=1.01)

    return fig


# ── Individual panel renderers ────────────────────────────────────────────

def _draw_confidence_line(ax, turns, emotions, confidences):
    """Line chart: confidence over conversation turns."""
    ax.set_facecolor("#2a2a3e")

    if len(turns) == 1:
        turns = [turns[0] - 0.5, turns[0], turns[0] + 0.5]
        confidences = [confidences[0]] * 3
        emotions_plot = [emotions[0]] * 3
    else:
        emotions_plot = emotions

    # Smooth line with gradient fill
    ax.plot(turns, confidences[:len(turns)], color="#7f8cff",
            linewidth=2.5, zorder=3)
    ax.fill_between(turns, confidences[:len(turns)], alpha=0.25,
                    color="#7f8cff")

    # Colour each point by emotion
    for i, (t, c, e) in enumerate(zip(turns, confidences, emotions_plot)):
        color = EMOTION_COLORS.get(e, "#9e9e9e")
        ax.scatter(t, c, color=color, s=80, zorder=5,
                   edgecolors="white", linewidths=0.8)

    ax.set_xlabel("Conversation Turn", fontsize=10)
    ax.set_ylabel("Confidence", fontsize=10)
    ax.set_title("Emotion Confidence Over Time", fontsize=11, pad=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, axis="y")
    ax.set_xticks(turns)


def _draw_emotion_timeline(ax, turns, emotions, confidences):
    """Scatter plot: emotion category vs turn, sized by confidence."""
    ax.set_facecolor("#2a2a3e")

    for t, e, c in zip(turns, emotions, confidences):
        y = EMOTION_ORDER.index(e) if e in EMOTION_ORDER else 4
        color = EMOTION_COLORS.get(e, "#9e9e9e")
        ax.scatter(t, y, color=color, s=c * 300, alpha=0.85,
                   edgecolors="white", linewidths=0.7, zorder=4)
        ax.text(t, y + 0.28, e[:5], ha="center", fontsize=7,
                color="white", alpha=0.8)

    ax.set_yticks(range(len(EMOTION_ORDER)))
    ax.set_yticklabels(EMOTION_ORDER, fontsize=9)
    ax.set_xlabel("Conversation Turn", fontsize=10)
    ax.set_title("Emotion Category Timeline", fontsize=11, pad=8)
    ax.set_ylim(-0.8, len(EMOTION_ORDER) - 0.2)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_xticks(turns)


def _draw_frequency_bars(ax, emotions):
    """Horizontal bar chart: emotion frequency."""
    ax.set_facecolor("#2a2a3e")

    from collections import Counter
    counts = Counter(emotions)
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [EMOTION_COLORS.get(e, "#9e9e9e") for e in labels]

    bars = ax.barh(labels, values, color=colors, edgecolor="#1e1e2e",
                   height=0.65)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10, color="white")

    ax.set_xlabel("Frequency", fontsize=10)
    ax.set_title("Emotion Frequency", fontsize=11, pad=8)
    ax.set_xlim(0, max(values) + 1.5)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.4)


def _draw_valence_trend(ax, turns, emotions):
    """Area chart: mood valence over turns (positive = better mood)."""
    ax.set_facecolor("#2a2a3e")

    valences = [VALENCE.get(e, 1) for e in emotions]

    if len(turns) == 1:
        plot_turns = [turns[0] - 0.5, turns[0], turns[0] + 0.5]
        plot_vals  = [valences[0]] * 3
    else:
        plot_turns = turns
        plot_vals  = valences

    # Colour the fill: green above midline (2), red below
    midline = 2
    ax.axhline(midline, color="#555575", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.plot(plot_turns, plot_vals, color="#c8c8ff",
            linewidth=2.2, zorder=3)

    # Fill above/below midline separately
    vals_arr  = np.array(plot_vals, dtype=float)
    turns_arr = np.array(plot_turns, dtype=float)
    ax.fill_between(turns_arr, vals_arr, midline,
                    where=(vals_arr >= midline), alpha=0.35,
                    color="#5cb85c", label="Better mood")
    ax.fill_between(turns_arr, vals_arr, midline,
                    where=(vals_arr < midline), alpha=0.35,
                    color="#e85d4a", label="Lower mood")

    # Scatter points
    for t, v, e in zip(plot_turns, plot_vals, emotions):
        color = EMOTION_COLORS.get(e, "#9e9e9e")
        ax.scatter(t, v, color=color, s=75, zorder=5,
                   edgecolors="white", linewidths=0.8)

    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["Very Low", "Low", "Neutral", "Good", "Great"], fontsize=8)
    ax.set_xlabel("Conversation Turn", fontsize=10)
    ax.set_title("Mood Valence Trend", fontsize=11, pad=8)
    ax.set_ylim(-0.5, 4.8)
    ax.set_xticks(turns)
    ax.legend(fontsize=8, loc="upper left",
              facecolor="#2a2a3e", edgecolor="#44445a",
              labelcolor="white")
    ax.grid(True, axis="y", alpha=0.4)


# ── Lightweight single chart for "live" sidebar ───────────────────────────

def build_mini_confidence_chart(emotion_history: List[Dict]) -> plt.Figure:
    """
    Small single-panel confidence chart.
    Suitable for embedding in the Chat tab sidebar.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#1e1e2e")

    if not emotion_history:
        ax.text(0.5, 0.5, "Start chatting to see your emotional trend",
                ha="center", va="center", fontsize=11, color="#7f8cff",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    turns       = [e["turn"] for e in emotion_history]
    emotions    = [e["emotion"] for e in emotion_history]
    confidences = [e["confidence"] for e in emotion_history]

    _draw_confidence_line(ax, turns, emotions, confidences)
    ax.set_title("Live Emotion Confidence", fontsize=10, pad=6)
    fig.tight_layout(pad=1.2)
    return fig


# ========================
# Testing
# ========================
if __name__ == "__main__":
    import os

    sample = [
        {"turn": 1, "emotion": "sad",        "confidence": 0.85, "valence": -1},
        {"turn": 2, "emotion": "anxious",     "confidence": 0.78, "valence": -1.5},
        {"turn": 3, "emotion": "stressed",    "confidence": 0.82, "valence": -1.5},
        {"turn": 4, "emotion": "overwhelmed", "confidence": 0.70, "valence": -1},
        {"turn": 5, "emotion": "neutral",     "confidence": 0.60, "valence": 0},
        {"turn": 6, "emotion": "hopeful",     "confidence": 0.72, "valence": 1},
    ]

    fig = build_emotion_dashboard(sample)
    out = "/tmp/emotion_dashboard_test.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"✅ Dashboard saved → {out}")
    plt.close(fig)

    fig2 = build_mini_confidence_chart(sample)
    out2 = "/tmp/mini_chart_test.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"✅ Mini chart saved → {out2}")
    plt.close(fig2)
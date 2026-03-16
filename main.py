"""
Mental Health Chatbot - Gradio UI (PHASE 3 - VISUAL UPGRADE)
Changes vs Phase 2:
  - Memes display inline as images (no external links, no page redirect)
  - Quotes display as beautiful visual PIL-generated cards
  - Live mini emotion chart updates after every message
  - Full 4-panel analytics dashboard in Analytics tab
  - All Phase 2 features preserved (explainability, comparison, stats)
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.append('/content/drive/MyDrive/mental_health_chatbot/src')

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/content/drive/MyDrive/mental_health_chatbot"
INTENTS_PATH = f"{PROJECT_ROOT}/data/intents.json"
MEMES_PATH   = f"{PROJECT_ROOT}/data/memes.json"
QUOTES_PATH  = f"{PROJECT_ROOT}/data/quotes.json"
MEMES_DIR    = f"{PROJECT_ROOT}/data/memes"

from chatbot import MentalHealthChatbot
from emotion_chart import build_emotion_dashboard, build_mini_confidence_chart

print("Initializing chatbot...")
chatbot = MentalHealthChatbot(
    intents_path=INTENTS_PATH,
    memes_path=MEMES_PATH,
    quotes_path=QUOTES_PATH
)
chatbot.meme_suggester.memes_dir = MEMES_DIR
print("Chatbot ready!")


# ─────────────────────────────────────────────────────────────────────────
# Callback functions
# ─────────────────────────────────────────────────────────────────────────

def submit_message(message, history):
    """
    Handle a chat message.
    Returns: [chatbot_history, cleared_input, meme_img, quote_img, live_chart]
    
    KEY RULE:
      - chatbot.last_meme_image_path  is set by _handle_meme_response  (or stays None)
      - chatbot.last_quote_card_path  is set by _handle_quote_response (or stays None)
      - Both are reset to None at the START of every turn so stale images don't persist
      - Image panels only update when a new path is produced this turn
    """
    if not message or not message.strip():
        return history, "", gr.update(), gr.update(), build_mini_confidence_chart([])

    # ── Reset image state BEFORE calling chat ────────────────────────────
    # This ensures we can distinguish "new image this turn" vs "no image this turn"
    chatbot.last_meme_image_path  = None
    chatbot.last_quote_card_path  = None

    # ── Run the full pipeline ─────────────────────────────────────────────
    bot_text = chatbot.chat(message.strip())

    # ── Append to chat history (ONLY the text response) ──────────────────
    history = history + [[message, bot_text]]

    # ── Read image outputs set by the pipeline ────────────────────────────
    meme_path  = chatbot.last_meme_image_path   # str path or None
    quote_path = chatbot.last_quote_card_path   # str path or None

    # ── Update live emotion chart ─────────────────────────────────────────
    mini_fig = build_mini_confidence_chart(
        getattr(chatbot.trend_tracker, "emotion_history", [])
    )

    # ── Build Gradio updates ──────────────────────────────────────────────
    # gr.update(value=path)  → shows the image
    # gr.update(value=None)  → clears the panel (no image this turn)
    # We always send an explicit update so panels reflect current state
    meme_update  = gr.update(value=meme_path)
    quote_update = gr.update(value=quote_path)

    return history, "", meme_update, quote_update, mini_fig


def reset_all():
    chatbot.reset_conversation()
    # Return gr.update(value=None) to explicitly clear both image panels
    return None, gr.update(value=None), gr.update(value=None), build_mini_confidence_chart([])


def get_explainability():
    return chatbot.get_explainability()


def get_full_dashboard():
    return build_emotion_dashboard(chatbot.trend_tracker.emotion_history)


def get_analytics_text():
    return chatbot.analytics.format_summary_for_display()


def run_comparison(user_message):
    if not user_message or not user_message.strip():
        return "Please enter a message to compare."
    return chatbot.run_comparison(user_message.strip())


def get_stats():
    stats = chatbot.get_stats()
    lines = [
        f"Total messages       : {stats['total_messages']}",
        f"Conversation turns   : {stats['conversation_turns']}",
        f"Last emotion         : {stats['last_emotion']} "
          f"(Confidence: {stats['last_emotion_confidence']})",
        f"Waiting for meme     : {stats['waiting_for_meme']}",
        f"Waiting for quote    : {stats['waiting_for_quote']}",
        "",
        "--- Emotional Trend ---",
        f"Dominant emotion : {stats['trend'].get('dominant_emotion', 'N/A')}",
        f"Drift direction  : {stats['trend'].get('drift_direction', 'N/A')}",
        f"Mood stability   : {stats['trend'].get('mood_stability', 0):.0%}",
        "",
        "--- Session Analytics ---",
        f"Crisis detections: {stats['analytics']['crisis_detections']}",
        f"Avg response time: {stats['analytics']['avg_response_time_seconds']}s",
        f"RAG responses    : {stats['analytics']['rag_responses']}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Gradio Layout
# ─────────────────────────────────────────────────────────────────────────

CSS = """
.image-panel img {
    border-radius: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    width: 100%;
}
.gr-button { border-radius: 8px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Mental Health Support Chatbot", css=CSS) as demo:

    # Header
    gr.HTML("""
    <div style="text-align:center; padding:20px 0 10px 0;">
      <h1 style="font-size:2rem; margin-bottom:6px;">
        🧠 Mental Health Support Chatbot
      </h1>
      <p style="color:#888; font-size:0.95rem;">
        இது ஒரு மன ஆரோக்கிய ஆதரவு chatbot &nbsp;|&nbsp;
        This is a mental health support chatbot
      </p>
      <p style="font-size:0.82rem; color:#aaa; margin-top:4px;">
        <b>AI Features:</b> Emotion Detection · Confidence Scoring · Risk Classification ·
        Emotion-Weighted RAG · Trend Tracking · Explainability · Visual Analytics
      </p>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════ TAB 1: CHAT ══════════════════════════
        with gr.Tab("💬 Chat"):

            with gr.Row(equal_height=False):

                # Left column — chat interface
                with gr.Column(scale=3, min_width=400):
                    chatbot_ui = gr.Chatbot(
                        height=450,
                        label="Conversation",
                        bubble_full_width=False,
                    )
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder=(
                            "Type here... / "
                            "இங்கே உங்கள் செய்தியை தட்டச்சு செய்யவும்..."
                        ),
                        lines=2,
                    )
                    with gr.Row():
                        send_btn  = gr.Button("Send 💬",  variant="primary", scale=3)
                        reset_btn = gr.Button("Reset 🔄", variant="secondary", scale=1)

                    with gr.Row():
                        explain_btn = gr.Button("🔍 System Reasoning", size="sm")
                        stats_btn   = gr.Button("📊 Quick Stats",       size="sm")

                    explain_out = gr.Textbox(
                        label="System Explainability",
                        lines=10, visible=False, interactive=False
                    )
                    stats_out = gr.Textbox(
                        label="Stats",
                        lines=10, visible=False, interactive=False
                    )

                # Right column — live visuals
                with gr.Column(scale=2, min_width=260):
                    gr.Markdown("**Live Emotion Chart**")
                    live_chart = gr.Plot(
                        label="Emotion Confidence",
                        show_label=False,
                    )

                    gr.Markdown("**Meme** *(appears when you accept a meme offer)*")
                    meme_img = gr.Image(
                        label="Meme",
                        type="filepath",
                        height=200,
                        show_download_button=False,
                        elem_classes=["image-panel"],
                    )

                    gr.Markdown("**Quote Card** *(appears when you accept a quote offer)*")
                    quote_img = gr.Image(
                        label="Quote Card",
                        type="filepath",
                        height=200,
                        show_download_button=False,
                        elem_classes=["image-panel"],
                    )

            # Example prompts
            gr.Examples(
                examples=[
                    ["I'm feeling very stressed about my exams"],
                    ["இன்று எனக்கு மிகவும் சோகமாக இருக்கிறது"],
                    ["I feel lonely and have no one to talk to"],
                    ["என்னால் தூங்க முடியவில்லை, மிகவும் பதட்டமாக இருக்கிறது"],
                    ["I'm overwhelmed with everything happening"],
                    ["நான் மிகவும் கோபமாக இருக்கிறேன்"],
                ],
                inputs=msg_input,
            )

            # Wire up buttons
            send_btn.click(
                fn=submit_message,
                inputs=[msg_input, chatbot_ui],
                outputs=[chatbot_ui, msg_input, meme_img, quote_img, live_chart],
            )
            msg_input.submit(
                fn=submit_message,
                inputs=[msg_input, chatbot_ui],
                outputs=[chatbot_ui, msg_input, meme_img, quote_img, live_chart],
            )
            reset_btn.click(
                fn=reset_all,
                outputs=[chatbot_ui, meme_img, quote_img, live_chart],
            )
            explain_btn.click(
                fn=get_explainability, outputs=explain_out
            ).then(fn=lambda: gr.update(visible=True), outputs=explain_out)
            stats_btn.click(
                fn=get_stats, outputs=stats_out
            ).then(fn=lambda: gr.update(visible=True), outputs=stats_out)

        # ════════════════ TAB 2: ANALYTICS DASHBOARD ═════════════════
        with gr.Tab("📈 Analytics & Trends"):
            gr.Markdown(
                """
                ### Emotional Analytics Dashboard
                Click **Refresh** after a few conversation turns to see your emotional journey.
                The dashboard shows 4 charts:
                confidence over time · category timeline · frequency · mood valence trend.
                """
            )
            refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")

            dashboard_plot = gr.Plot(label="4-Panel Analytics Dashboard")
            analytics_text = gr.Textbox(
                label="Session Statistics", lines=16, interactive=False
            )

            refresh_btn.click(
                fn=lambda: (get_full_dashboard(), get_analytics_text()),
                outputs=[dashboard_plot, analytics_text],
            )

        # ════════════════ TAB 3: RAG vs NON-RAG ══════════════════════
        with gr.Tab("⚗️ RAG vs Non-RAG"):
            gr.Markdown(
                """
                ### Comparison Mode
                Enter any message to compare:
                - **Mode 1** — LLM with no retrieved context
                - **Mode 2** — LLM with emotion-weighted RAG context

                Each response is scored on emotional alignment, length quality,
                and context specificity.
                """
            )
            cmp_input  = gr.Textbox(
                label="Message to compare",
                placeholder="e.g. I'm feeling really anxious about my future",
                lines=2,
            )
            cmp_btn    = gr.Button("⚗️ Run Comparison", variant="primary")
            cmp_output = gr.Textbox(
                label="Comparison Report", lines=28, interactive=False
            )
            cmp_btn.click(fn=run_comparison, inputs=cmp_input, outputs=cmp_output)

    # Footer
    gr.HTML("""
    <div style="text-align:center; padding:16px 0; color:#888; font-size:0.82rem;">
      <b>⚠️ Disclaimer:</b> This chatbot is a supportive companion,
      NOT a substitute for professional mental health services.<br>
      <b>Crisis Helplines (India):</b>
      AASRA: 91-9820466726 &nbsp;·&nbsp;
      Sneha (Chennai): 044-24640050 &nbsp;·&nbsp;
      Emergency: 112<br><br>
      <i>CS Minor Project &nbsp;|&nbsp;
      Stack: Python · ChromaDB · Groq API · Gradio · sentence-transformers · matplotlib · Pillow</i>
    </div>
    """)


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LAUNCHING MENTAL HEALTH CHATBOT  (PHASE 3)")
    print("=" * 60 + "\n")
    demo.launch(share=True, debug=True, show_error=True)
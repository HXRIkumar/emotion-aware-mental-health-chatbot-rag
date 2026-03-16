"""
Meme Suggestion Module (FINAL FIX)
- Generates PIL-based meme cards on the fly (no external PNG files needed)
- image_path is ALWAYS populated when a meme is retrieved
- format_meme_message() returns ONLY a short chat acknowledgement (no full caption in chat)
- All original logic preserved exactly
"""

import os
import textwrap
from typing import List, Dict, Optional

# ── Meme card visual themes ────────────────────────────────────────────────
MEME_THEMES = {
    "sad":        {"bg": "#1a1a2e", "bar": "#e94560", "text": "#eaeaea", "sub": "#a8a8c8"},
    "lonely":     {"bg": "#1b1b2f", "bar": "#e2b96f", "text": "#f5f0e0", "sub": "#c4a96a"},
    "anxious":    {"bg": "#0f3460", "bar": "#533483", "text": "#e8e8ff", "sub": "#a0b4ff"},
    "stressed":   {"bg": "#16213e", "bar": "#0f9b8e", "text": "#e0f7f4", "sub": "#7fd8d0"},
    "overwhelmed":{"bg": "#1f1f1f", "bar": "#6c63ff", "text": "#f0eeff", "sub": "#b0a8ff"},
    "angry":      {"bg": "#1a0a00", "bar": "#ff6b35", "text": "#fff0e8", "sub": "#ffb08a"},
    "neutral":    {"bg": "#1e1e2e", "bar": "#7c7c9c", "text": "#e0e0f0", "sub": "#9898b8"},
}
DEFAULT_THEME = MEME_THEMES["neutral"]


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def generate_meme_card(caption_en: str, caption_ta: str,
                       emotion: str = "neutral",
                       tmp_dir: str = "/tmp/meme_cards") -> Optional[str]:
    """
    Generate a meme card PNG using PIL and return its file path.
    Falls back gracefully if PIL is unavailable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    os.makedirs(tmp_dir, exist_ok=True)

    theme = MEME_THEMES.get(emotion, DEFAULT_THEME)
    W, H = 520, 300

    img  = Image.new("RGB", (W, H), _hex_to_rgb(theme["bg"]))
    draw = ImageDraw.Draw(img)

    # Top accent bar
    draw.rectangle([0, 0, W, 8], fill=_hex_to_rgb(theme["bar"]))
    # Bottom accent bar
    draw.rectangle([0, H - 8, W, H], fill=_hex_to_rgb(theme["bar"]))

    # Emoji / icon area
    draw.rectangle([20, 20, 70, 70], fill=_hex_to_rgb(theme["bar"]))
    try:
        icon_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        icon_font = ImageFont.load_default()
    draw.text((24, 26), ":)", font=icon_font, fill=_hex_to_rgb(theme["text"]))

    # "MEME" label
    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        body_font  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      18)
        sub_font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      13)
    except Exception:
        label_font = body_font = sub_font = ImageFont.load_default()

    draw.text((80, 28), "MEME CARD", font=label_font, fill=_hex_to_rgb(theme["bar"]))
    draw.text((80, 46), emotion.upper(), font=sub_font, fill=_hex_to_rgb(theme["sub"]))

    # Main caption (English)
    wrapped = textwrap.wrap(caption_en, width=44)
    y = 90
    for line in wrapped[:4]:
        draw.text((30, y), line, font=body_font, fill=_hex_to_rgb(theme["text"]))
        y += 28

    # Tamil caption (smaller, below)
    if caption_ta:
        draw.line([(30, y + 10), (W - 30, y + 10)], fill=_hex_to_rgb(theme["bar"]), width=1)
        try:
            ta_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            ta_font = ImageFont.load_default()
        ta_wrapped = textwrap.wrap(caption_ta, width=56)
        ty = y + 22
        for line in ta_wrapped[:2]:
            draw.text((30, ty), line, font=ta_font, fill=_hex_to_rgb(theme["sub"]))
            ty += 18

    # Save
    safe_emotion = emotion.replace(" ", "_")
    out_path = os.path.join(tmp_dir, f"meme_{safe_emotion}.png")
    img.save(out_path, "PNG")
    return out_path


class MemeSuggester:
    """
    Suggests appropriate memes based on detected emotion and user message.
    Works with RAG engine to retrieve semantically relevant memes.
    FINAL: generate_meme_card() produces a PIL image — no external PNG files needed.
    """

    def __init__(self, rag_engine, memes_dir: str = "data/memes"):
        self.rag_engine = rag_engine
        self.memes_dir  = memes_dir  # kept for compatibility, not required

    # ── Unchanged decision methods ────────────────────────────────────

    def should_suggest_meme(self, emotion: str) -> bool:
        return emotion in ["sad", "lonely", "anxious", "stressed", "overwhelmed", "angry"]

    def create_meme_offer_message(self, lang: str = "en") -> str:
        if lang == "ta":
            return (
                "\n\nஒரு சின்ன meme உங்களுக்கு அனுப்பலாமா? "
                "கொஞ்சம் இலேசாக உணரலாம். 🙂\n"
                "('yes', 'meme' அல்லது 'ஆம்' என்று type பண்ணுங்க)"
            )
        return (
            "\n\nWould you like me to send you a small meme? "
            "It might help lighten things a bit. 🙂\n"
            "(Type 'yes' or 'meme' if you'd like one)"
        )

    def is_meme_confirmation(self, message: str) -> bool:
        lower = message.lower().strip()
        for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "meme", "send", "please", "pls"]:
            if w in lower:
                return True
        for w in ["ஆம்", "ஆம", "சரி", "மீம்", "அனுப்பு"]:
            if w in message:
                return True
        return False

    def is_meme_rejection(self, message: str) -> bool:
        lower = message.lower().strip()
        for w in ["no", "nope", "nah", "not now", "maybe later", "skip", "don't want"]:
            if w in lower:
                return True
        for w in ["வேண்டாம்", "இல்லை", "பிறகு"]:
            if w in message:
                return True
        return False

    # ── Core retrieval — NOW generates PIL card ───────────────────────

    def get_meme(
        self,
        user_message: str,
        emotion: Optional[str] = None,
        k: int = 1
    ) -> Optional[Dict[str, str]]:
        """
        Retrieve a meme from RAG and generate a visual card PNG.

        Returns dict with:
            caption_en, caption_ta, emotion_tags  (from RAG)
            image_path  — absolute path to generated PNG card (never None on success)
        """
        memes = self.rag_engine.retrieve_meme(user_message, k=k)
        if not memes:
            return None

        meme = memes[0].copy()
        caption_en = meme.get("caption_en", "Here's something to lighten your day!")
        caption_ta = meme.get("caption_ta", "")

        # Generate PIL card — this always works, no external files needed
        image_path = generate_meme_card(
            caption_en=caption_en,
            caption_ta=caption_ta,
            emotion=emotion or "neutral"
        )

        meme["image_path"] = image_path
        return meme

    def format_meme_message(self, meme: Dict[str, str], lang: str = "en") -> str:
        """
        Returns a SHORT chat acknowledgement only.
        The actual image is displayed in the Meme panel via image_path.
        Caption is intentionally NOT included in chat to keep it clean.
        """
        if lang == "ta":
            return "😊 மேலே வலதுபுறம் உள்ள Meme panel-ல் பாருங்கள்! 💙"
        return "😊 Check the Meme panel on the right side! 💙"
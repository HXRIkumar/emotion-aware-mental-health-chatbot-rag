"""
Quote Card Generator
Dynamically generates beautiful quote card images using PIL.
No external image storage needed — generated on the fly.
"""

import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

# ── Color themes per emotion ───────────────────────────────────────────────
EMOTION_THEMES = {
    "sad": {
        "bg": "#1a1a2e",        # deep navy
        "top_bar": "#e94560",   # rose
        "text": "#eaeaea",
        "author": "#a8a8c8",
        "quote_mark": "#e94560",
    },
    "anxious": {
        "bg": "#0f3460",        # deep blue
        "top_bar": "#533483",   # purple
        "text": "#e8e8ff",
        "author": "#a0b4ff",
        "quote_mark": "#533483",
    },
    "stressed": {
        "bg": "#16213e",        # dark blue
        "top_bar": "#0f9b8e",   # teal
        "text": "#e0f7f4",
        "author": "#7fd8d0",
        "quote_mark": "#0f9b8e",
    },
    "lonely": {
        "bg": "#1b1b2f",        # dark indigo
        "top_bar": "#e2b96f",   # warm gold
        "text": "#f5f0e0",
        "author": "#c4a96a",
        "quote_mark": "#e2b96f",
    },
    "overwhelmed": {
        "bg": "#1f1f1f",        # near black
        "top_bar": "#6c63ff",   # violet
        "text": "#f0eeff",
        "author": "#b0a8ff",
        "quote_mark": "#6c63ff",
    },
    "angry": {
        "bg": "#1a0a00",        # dark brown
        "top_bar": "#ff6b35",   # orange
        "text": "#fff0e8",
        "author": "#ffb08a",
        "quote_mark": "#ff6b35",
    },
    "hopeful": {
        "bg": "#0d2b1a",        # forest dark
        "top_bar": "#56c596",   # mint green
        "text": "#e8fff4",
        "author": "#8ae0b8",
        "quote_mark": "#56c596",
    },
    "neutral": {
        "bg": "#1e1e2e",        # dark slate
        "top_bar": "#7f8cff",   # soft blue
        "text": "#e8e8ff",
        "author": "#a0a8d0",
        "quote_mark": "#7f8cff",
    },
    "happy": {
        "bg": "#1a2600",        # dark olive
        "top_bar": "#f9ca24",   # sunny yellow
        "text": "#fffbe0",
        "author": "#d4c060",
        "quote_mark": "#f9ca24",
    },
}

DEFAULT_THEME = EMOTION_THEMES["neutral"]

W, H = 600, 380


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _get_fonts():
    """Load fonts, fall back to default if unavailable."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    def try_font(path, size):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return None

    quote_font   = try_font(font_paths[0], 22) or try_font(font_paths[2], 22) or ImageFont.load_default()
    author_font  = try_font(font_paths[1], 17) or try_font(font_paths[3], 17) or ImageFont.load_default()
    label_font   = try_font(font_paths[3], 13) or ImageFont.load_default()
    big_font     = try_font(font_paths[0], 72) or ImageFont.load_default()

    return quote_font, author_font, label_font, big_font


def generate_quote_card(
    quote_text: str,
    author: str,
    emotion: str = "neutral",
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Generate a beautiful quote card image.

    Args:
        quote_text (str): The quote text (English or Tamil)
        author (str): Author name
        emotion (str): Emotion label — determines color theme
        save_path (str, optional): If given, saves PNG to this path

    Returns:
        PIL.Image.Image: The generated card
    """
    theme = EMOTION_THEMES.get(emotion, DEFAULT_THEME)

    bg_rgb      = _hex_to_rgb(theme["bg"])
    bar_rgb     = _hex_to_rgb(theme["top_bar"])
    text_rgb    = _hex_to_rgb(theme["text"])
    author_rgb  = _hex_to_rgb(theme["author"])
    qmark_rgb   = _hex_to_rgb(theme["quote_mark"])

    img  = Image.new("RGB", (W, H), bg_rgb)
    draw = ImageDraw.Draw(img)

    quote_font, author_font, label_font, big_font = _get_fonts()

    # ── Gradient top bar (thick) ──────────────────────────────────────
    for i in range(8):
        opacity = 1.0 - i * 0.12
        r = int(bar_rgb[0] * opacity + bg_rgb[0] * (1 - opacity))
        g = int(bar_rgb[1] * opacity + bg_rgb[1] * (1 - opacity))
        b = int(bar_rgb[2] * opacity + bg_rgb[2] * (1 - opacity))
        draw.rectangle([0, i*5, W, i*5+5], fill=(r, g, b))

    # ── Left accent bar ───────────────────────────────────────────────
    draw.rectangle([0, 40, 6, H], fill=bar_rgb)

    # ── Big decorative quote mark ─────────────────────────────────────
    # Faint background quotation mark
    qmark_faint = (*qmark_rgb, 30)   # need RGBA for transparency trick
    # Just draw a large faded " using normal mode
    draw.text((W - 110, H - 130), "\u201C", font=big_font,
              fill=(min(qmark_rgb[0]+60, 255),
                    min(qmark_rgb[1]+60, 255),
                    min(qmark_rgb[2]+60, 255)))

    # ── Small opening quote mark ──────────────────────────────────────
    draw.text((24, 48), "\u201C", font=big_font, fill=qmark_rgb)

    # ── Wrap quote text ───────────────────────────────────────────────
    max_chars = 42
    wrapped = textwrap.fill(quote_text, width=max_chars)
    lines = wrapped.split("\n")

    # Start below the big quote mark
    y = 120
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=quote_font)
        tw = bbox[2] - bbox[0]
        x = max(28, (W - tw) // 2)
        draw.text((x, y), line, font=quote_font, fill=text_rgb)
        y += 32

    # ── Divider line ──────────────────────────────────────────────────
    y += 10
    draw.rectangle([40, y, W - 40, y + 2], fill=bar_rgb)
    y += 14

    # ── Author ────────────────────────────────────────────────────────
    author_line = f"— {author}"
    bbox = draw.textbbox((0, 0), author_line, font=author_font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, y), author_line, font=author_font, fill=author_rgb)

    # ── Bottom label ──────────────────────────────────────────────────
    draw.text((20, H - 22), "💙 Mental Health Chatbot  •  You are not alone",
              font=label_font, fill=author_rgb)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path, "PNG", optimize=True)

    return img


# ── Convenience wrapper ────────────────────────────────────────────────────

def generate_quote_card_for_gradio(
    quote_text: str,
    author: str,
    emotion: str = "neutral",
    tmp_dir: str = "/tmp/quote_cards"
) -> str:
    """
    Generate quote card and save to a temp file.
    Returns the file path — suitable for gr.Image(value=path).

    Args:
        quote_text, author, emotion: as above
        tmp_dir: directory to save temp PNG

    Returns:
        str: Absolute path to generated PNG
    """
    os.makedirs(tmp_dir, exist_ok=True)
    # Use emotion as part of filename so repeated same-emotion reuses file
    path = os.path.join(tmp_dir, f"quote_{emotion}.png")
    generate_quote_card(quote_text, author, emotion, save_path=path)
    return path


# ========================
# Testing
# ========================
if __name__ == "__main__":
    import os

    test_quotes = [
        ("Even the darkest night will end and the sun will rise.", "Victor Hugo", "sad"),
        ("You don't have to control your thoughts. You just have to stop letting them control you.", "Dan Millman", "anxious"),
        ("Almost everything will work again if you unplug it for a few minutes, including you.", "Anne Lamott", "stressed"),
        ("Hope is being able to see that there is light despite all of the darkness.", "Desmond Tutu", "hopeful"),
    ]

    out_dir = "/tmp/quote_test"
    os.makedirs(out_dir, exist_ok=True)

    for quote, author, emotion in test_quotes:
        path = os.path.join(out_dir, f"test_{emotion}.png")
        img = generate_quote_card(quote, author, emotion, save_path=path)
        print(f"✅ {emotion}: {img.size} → {path}")

    print("\nAll quote cards generated!")
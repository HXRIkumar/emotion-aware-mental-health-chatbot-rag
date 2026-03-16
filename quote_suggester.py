"""
Quote Suggester Module (UPGRADED)
Provides inspirational mental health quotes based on detected emotions.
UPGRADE: Added generate_card() which returns a PIL Image path via quote_card_generator.
All original logic preserved exactly.
"""

import json
import random
from typing import Dict, List, Optional


class QuoteSuggester:
    """
    Suggests inspirational quotes based on user's emotional state.
    Now also generates visual quote cards.
    """

    def __init__(self, quotes_path: str):
        self.quotes_path = quotes_path
        self.quotes = self._load_quotes()
        self.quotes_by_emotion = self._organize_by_emotion()
        print(f"✅ Quote suggester ready ({len(self.quotes)} quotes loaded)")

    # ── Original private methods (unchanged) ──────────────────────────

    def _load_quotes(self) -> List[Dict]:
        try:
            with open(self.quotes_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Quotes file not found: {self.quotes_path}")
            return []
        except Exception as e:
            print(f"⚠️  Error loading quotes: {e}")
            return []

    def _organize_by_emotion(self) -> Dict[str, List[Dict]]:
        organized = {}
        for quote in self.quotes:
            emotion = quote.get('emotion', 'neutral')
            if emotion not in organized:
                organized[emotion] = []
            organized[emotion].append(quote)
        return organized

    # ── Original public methods (unchanged) ───────────────────────────

    def should_suggest_quote(self, emotion: str, message_count: int = 0) -> bool:
        negative_emotions = ["sad", "anxious", "stressed", "lonely", "overwhelmed", "angry"]
        if emotion in negative_emotions and message_count >= 2:
            return True
        return False

    def get_quote(self, emotion: str, lang: str = "en") -> Optional[Dict[str, str]]:
        emotion_quotes = self.quotes_by_emotion.get(emotion, [])
        if not emotion_quotes:
            emotion_quotes = self.quotes_by_emotion.get('neutral', [])
        if not emotion_quotes:
            return None
        quote = random.choice(emotion_quotes)
        return {
            'quote_en': quote.get('quote_en', ''),
            'quote_ta': quote.get('quote_ta', ''),
            'author':   quote.get('author', 'Unknown'),
            'emotion':  quote.get('emotion', emotion)
        }

    def format_quote_message(self, quote: Dict[str, str], lang: str = "en") -> str:
        if lang == "ta":
            quote_text = quote.get('quote_ta', quote.get('quote_en', ''))
            prefix = "💭 உங்களுக்கான ஒரு சிந்தனை:\n\n"
            suffix = f"\n\n— {quote.get('author', 'Unknown')}"
        else:
            quote_text = quote.get('quote_en', '')
            prefix = "💭 A thought for you:\n\n"
            suffix = f"\n\n— {quote.get('author', 'Unknown')}"
        return f"{prefix}\"{quote_text}\"{suffix}"

    def create_quote_offer_message(self, lang: str = "en") -> str:
        if lang == "ta":
            return (
                "\n\n💭 உங்களுக்கு ஊக்கமளிக்கும் ஒரு quote பகிரலாமா? "
                "இது சிறிது ஊக்கம் தரலாம்.\n"
                "(விரும்பினால் 'ஆம்', 'quote', அல்லது 'yes' என்று type செய்யுங்கள்)"
            )
        else:
            return (
                "\n\n💭 Would you like me to share an inspirational quote? "
                "It might provide some encouragement.\n"
                "(Type 'yes' or 'quote' if you'd like one)"
            )

    def is_quote_confirmation(self, message: str) -> bool:
        lower_msg = message.lower().strip()
        en_confirmations = ["yes", "yeah", "yep", "sure", "ok", "okay",
                            "quote", "send", "please", "pls", "share"]
        ta_confirmations = ["ஆம்", "ஆம", "சரி", "quote", "பகிர்"]
        for word in en_confirmations:
            if word in lower_msg:
                return True
        for word in ta_confirmations:
            if word in message:
                return True
        return False

    def is_quote_rejection(self, message: str) -> bool:
        lower_msg = message.lower().strip()
        en_rejections = ["no", "nope", "nah", "not now", "maybe later",
                         "skip", "don't want", "no thanks"]
        ta_rejections = ["வேண்டாம்", "இல்லை", "பிறகு"]
        for word in en_rejections:
            if word in lower_msg:
                return True
        for word in ta_rejections:
            if word in message:
                return True
        return False

    # ── NEW: visual quote card ────────────────────────────────────────

    def generate_card(
        self,
        quote: Dict[str, str],
        lang: str = "en",
        tmp_dir: str = "/tmp/quote_cards"
    ) -> Optional[str]:
        """
        Generate a visual quote card PNG and return its file path.

        Args:
            quote (Dict): Quote dict from get_quote()
            lang (str): "en" or "ta"
            tmp_dir (str): Directory to save temp PNG

        Returns:
            str | None: Absolute path to PNG, or None on failure
        """
        try:
            from quote_card_generator import generate_quote_card_for_gradio
            quote_text = (quote.get('quote_ta', '') if lang == "ta"
                          else quote.get('quote_en', ''))
            if not quote_text:
                quote_text = quote.get('quote_en', '')
            author  = quote.get('author', 'Unknown')
            emotion = quote.get('emotion', 'neutral')
            return generate_quote_card_for_gradio(quote_text, author, emotion, tmp_dir)
        except Exception as e:
            print(f"⚠️  Quote card generation failed: {e}")
            return None
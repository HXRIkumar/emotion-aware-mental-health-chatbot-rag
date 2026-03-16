"""
Emotion Detection Module
Detects emotion from user message using retrieved intents.
UPGRADED: Now returns a confidence score alongside the emotion label.
"""

from typing import List, Dict, Optional, Tuple


class EmotionDetector:
    """
    Detects emotions from user messages using RAG-retrieved intents.
    Returns both an emotion label and a confidence score.
    """

    EMOTIONS = [
        "sad", "anxious", "stressed", "lonely",
        "angry", "overwhelmed", "hopeful", "neutral", "happy"
    ]

    NEGATIVE_EMOTIONS = [
        "sad", "lonely", "anxious", "stressed", "overwhelmed", "angry"
    ]

    def __init__(self):
        pass

    # ── Core detection ────────────────────────────────────────────────

    def detect_from_retrieved_intents(
        self,
        retrieved_responses: List[Dict[str, str]]
    ) -> str:
        """
        Detect emotion from RAG-retrieved support responses (backward-compatible).

        Args:
            retrieved_responses: List of support responses from RAG

        Returns:
            str: Detected emotion label
        """
        emotion, _ = self.detect_with_confidence(retrieved_responses)
        return emotion

    def detect_with_confidence(
        self,
        retrieved_responses: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        """
        Detect emotion AND compute a confidence score.

        Confidence is based on agreement across the top-k retrieved intents.
        If all k results agree → high confidence.
        If they disagree → lower confidence.

        Args:
            retrieved_responses: List of support responses (each has "emotion" key)

        Returns:
            Tuple[str, float]: (emotion_label, confidence_score 0.0–1.0)
        """
        if not retrieved_responses:
            return "neutral", 0.5

        # Count emotion votes from retrieved intents
        emotion_votes: Dict[str, int] = {}
        for resp in retrieved_responses:
            emo = resp.get("emotion", "neutral")
            if emo not in self.EMOTIONS:
                emo = "neutral"
            emotion_votes[emo] = emotion_votes.get(emo, 0) + 1

        total = len(retrieved_responses)
        top_emotion = max(emotion_votes, key=lambda e: emotion_votes[e])
        top_count = emotion_votes[top_emotion]

        # Confidence = fraction of intents that agree on top emotion
        # Scaled slightly so even 1/1 = 0.70 (we are never 100% certain)
        raw_ratio = top_count / total          # 0.33 – 1.0
        confidence = round(0.50 + 0.45 * raw_ratio, 3)  # maps to 0.65 – 0.95

        return top_emotion, confidence

    # ── Helper methods (unchanged from original) ──────────────────────

    def is_negative_emotion(self, emotion: str) -> bool:
        return emotion in self.NEGATIVE_EMOTIONS

    def get_emotion_label(self, emotion: str, lang: str = "en") -> str:
        labels = {
            "sad":        {"en": "Sadness",      "ta": "சோகம்"},
            "anxious":    {"en": "Anxiety",       "ta": "பதட்டம்"},
            "stressed":   {"en": "Stress",        "ta": "மன அழுத்தம்"},
            "lonely":     {"en": "Loneliness",    "ta": "தனிமை"},
            "angry":      {"en": "Anger",         "ta": "கோபம்"},
            "overwhelmed":{"en": "Overwhelmed",   "ta": "மிகையான உணர்வு"},
            "hopeful":    {"en": "Hope",          "ta": "நம்பிக்கை"},
            "neutral":    {"en": "Neutral",       "ta": "நடுநிலை"},
            "happy":      {"en": "Happiness",     "ta": "மகிழ்ச்சி"},
        }
        return labels.get(emotion, {}).get(lang, emotion.capitalize())

    def get_emotion_emoji(self, emotion: str) -> str:
        emojis = {
            "sad": "😢", "anxious": "😰", "stressed": "😓",
            "lonely": "😔", "angry": "😠", "overwhelmed": "😵",
            "hopeful": "🌟", "neutral": "😐", "happy": "😊",
        }
        return emojis.get(emotion, "💭")

    def get_emotion_summary(
        self,
        emotion: str,
        lang: str = "en",
        include_emoji: bool = True,
        confidence: Optional[float] = None
    ) -> str:
        """
        Formatted emotion summary, optionally including confidence.

        Args:
            emotion: Emotion label
            lang: Language code
            include_emoji: Whether to prepend emoji
            confidence: Optional confidence score to display

        Returns:
            str: e.g. "😰 Anxiety (82%)"
        """
        label = self.get_emotion_label(emotion, lang)
        emoji = self.get_emotion_emoji(emotion) if include_emoji else ""
        conf_str = f" ({confidence:.0%})" if confidence is not None else ""

        parts = []
        if emoji:
            parts.append(emoji)
        parts.append(f"{label}{conf_str}")
        return " ".join(parts)


# ========================
# Testing
# ========================
if __name__ == "__main__":
    detector = EmotionDetector()

    # Unanimous agreement → high confidence
    unanimous = [
        {"text_en": "...", "emotion": "sad"},
        {"text_en": "...", "emotion": "sad"},
        {"text_en": "...", "emotion": "sad"},
    ]
    emo, conf = detector.detect_with_confidence(unanimous)
    print(f"Unanimous → {emo} ({conf:.0%})")   # expect sad ~95%

    # Partial agreement → medium confidence
    mixed = [
        {"text_en": "...", "emotion": "sad"},
        {"text_en": "...", "emotion": "anxious"},
        {"text_en": "...", "emotion": "sad"},
    ]
    emo, conf = detector.detect_with_confidence(mixed)
    print(f"Mixed     → {emo} ({conf:.0%})")   # expect sad ~80%

    # No agreement → low confidence
    split = [
        {"text_en": "...", "emotion": "sad"},
        {"text_en": "...", "emotion": "anxious"},
        {"text_en": "...", "emotion": "stressed"},
    ]
    emo, conf = detector.detect_with_confidence(split)
    print(f"Split     → {emo} ({conf:.0%})")   # expect ~65%

    # Summary with confidence
    print(detector.get_emotion_summary("anxious", confidence=0.82))
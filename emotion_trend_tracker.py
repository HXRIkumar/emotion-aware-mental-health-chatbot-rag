"""
Emotion Trend Tracker
Tracks emotional state across conversation turns.
Computes: dominant emotion, emotional drift, mood stability score.
"""

from collections import Counter
from typing import List, Dict, Optional, Tuple


# Assign numeric valence to emotions (negative → positive scale)
EMOTION_VALENCE = {
    "sad": -2,
    "angry": -2,
    "anxious": -1.5,
    "stressed": -1.5,
    "overwhelmed": -1,
    "lonely": -1,
    "neutral": 0,
    "hopeful": 1,
    "happy": 2,
}


class EmotionTrendTracker:
    """
    Tracks emotional states across conversation turns and computes trend metrics.
    """

    def __init__(self):
        self.emotion_history: List[Dict] = []  # [{turn, emotion, confidence, valence}]

    def add_emotion(self, emotion: str, confidence: float = 1.0, turn: Optional[int] = None):
        """
        Record an emotion for a given turn.

        Args:
            emotion (str): Detected emotion label
            confidence (float): Confidence score (0.0 – 1.0)
            turn (int, optional): Turn number; auto-increments if None
        """
        turn_num = turn if turn is not None else len(self.emotion_history) + 1
        valence = EMOTION_VALENCE.get(emotion, 0)
        self.emotion_history.append({
            "turn": turn_num,
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "valence": valence
        })

    def get_dominant_emotion(self) -> str:
        """
        Return the most frequently occurring emotion so far.

        Returns:
            str: Dominant emotion label, or 'neutral' if no history
        """
        if not self.emotion_history:
            return "neutral"
        counts = Counter(e["emotion"] for e in self.emotion_history)
        return counts.most_common(1)[0][0]

    def get_emotional_drift(self) -> float:
        """
        Compute emotional drift = change in valence from first to last turn.

        Positive drift → improving mood.
        Negative drift → worsening mood.

        Returns:
            float: Drift value
        """
        if len(self.emotion_history) < 2:
            return 0.0
        first_valence = self.emotion_history[0]["valence"]
        last_valence = self.emotion_history[-1]["valence"]
        return round(last_valence - first_valence, 2)

    def get_mood_stability(self) -> float:
        """
        Compute mood stability score (0.0 = very unstable, 1.0 = perfectly stable).

        Based on variance of valence values across turns.

        Returns:
            float: Stability score between 0.0 and 1.0
        """
        if len(self.emotion_history) < 2:
            return 1.0

        valences = [e["valence"] for e in self.emotion_history]
        mean = sum(valences) / len(valences)
        variance = sum((v - mean) ** 2 for v in valences) / len(valences)

        # Max possible variance given valence range [-2, 2] is 4.0
        stability = max(0.0, 1.0 - (variance / 4.0))
        return round(stability, 3)

    def get_trend_summary(self) -> Dict:
        """
        Return a full trend summary dictionary.

        Returns:
            Dict with keys: emotion_history, dominant_emotion,
                            emotional_drift, mood_stability, total_turns
        """
        drift = self.get_emotional_drift()
        drift_label = "improving" if drift > 0 else ("worsening" if drift < 0 else "stable")

        return {
            "total_turns": len(self.emotion_history),
            "emotion_history": self.emotion_history,
            "dominant_emotion": self.get_dominant_emotion(),
            "emotional_drift": drift,
            "drift_direction": drift_label,
            "mood_stability": self.get_mood_stability(),
        }

    def format_trend_for_display(self) -> str:
        """
        Return a human-readable trend summary string.

        Returns:
            str: Formatted trend summary
        """
        if not self.emotion_history:
            return "No emotional trend data yet."

        summary = self.get_trend_summary()
        history_str = " → ".join(
            f"{e['emotion']}({e['confidence']:.0%})"
            for e in self.emotion_history[-5:]  # Show last 5
        )

        return (
            f"📈 Emotional Trend (last {min(5, summary['total_turns'])} turns):\n"
            f"   Path: {history_str}\n"
            f"   Dominant Emotion: {summary['dominant_emotion']}\n"
            f"   Mood Drift: {summary['drift_direction']} ({summary['emotional_drift']:+.1f})\n"
            f"   Mood Stability: {summary['mood_stability']:.0%}"
        )

    def clear(self):
        """Reset emotion history."""
        self.emotion_history = []


# ========================
# Testing
# ========================
if __name__ == "__main__":
    tracker = EmotionTrendTracker()

    turns = [
        ("sad", 0.85),
        ("anxious", 0.78),
        ("stressed", 0.72),
        ("neutral", 0.60),
        ("hopeful", 0.65),
    ]

    for emotion, confidence in turns:
        tracker.add_emotion(emotion, confidence)

    print(tracker.format_trend_for_display())
    print("\nFull summary:")
    import json
    print(json.dumps(tracker.get_trend_summary(), indent=2))
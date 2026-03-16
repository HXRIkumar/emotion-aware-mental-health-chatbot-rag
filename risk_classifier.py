"""
Risk Level Classifier
Classifies conversation risk into LOW / MODERATE / HIGH.

HIGH  → crisis/self-harm phrases → bypass LLM, return helpline message
MODERATE → extreme emotional distress signals
LOW   → general stress or sadness
"""

from typing import Tuple

# HIGH risk: explicit self-harm / suicidal intent
HIGH_RISK_KEYWORDS = [
    "kill myself", "want to die", "wanna die", "end my life",
    "suicide", "suicidal", "hang myself", "jump off", "poison myself",
    "self harm", "self-harm", "cut myself", "hurt myself",
    "can't go on", "cant go on", "no reason to live",
    "better off dead", "end it all",
    # Tamil
    "தற்கொலை", "உயிர் விட", "தூக்கு", "குதிக்க",
]

# MODERATE risk: extreme distress without explicit self-harm
MODERATE_RISK_KEYWORDS = [
    "hopeless", "worthless", "nobody cares", "i give up",
    "can't take it", "falling apart", "breaking down",
    "everything is pointless", "i hate myself", "hate my life",
    "no one loves me", "i'm a burden", "disappear",
    # Tamil
    "வாழ்க்கை வேண்டாம்", "யாரும் இல்லை", "சோர்ந்து போனேன்",
]

# HIGH-risk emotions (used as a booster signal)
HIGH_RISK_EMOTIONS = ["sad", "overwhelmed", "lonely"]
MODERATE_RISK_EMOTIONS = ["anxious", "stressed", "angry"]


def classify_risk(
    message: str,
    emotion: str,
    confidence: float = 1.0
) -> Tuple[str, float, str]:
    """
    Classify the risk level of a user message.

    Args:
        message (str): User's input text
        emotion (str): Detected emotion label
        confidence (float): Emotion detection confidence (0.0 – 1.0)

    Returns:
        Tuple of (risk_level, risk_score, reason):
            risk_level (str): "HIGH", "MODERATE", or "LOW"
            risk_score (float): Numeric risk score 0.0 – 1.0
            reason (str): Short explanation
    """
    lower_msg = message.lower()

    # ── HIGH risk check ───────────────────────────────────────────────
    for kw in HIGH_RISK_KEYWORDS:
        if kw in lower_msg or kw in message:  # second check for Tamil
            return "HIGH", 1.0, f"Crisis keyword detected: '{kw}'"

    # ── MODERATE risk check ───────────────────────────────────────────
    moderate_hit = None
    for kw in MODERATE_RISK_KEYWORDS:
        if kw in lower_msg or kw in message:
            moderate_hit = kw
            break

    if moderate_hit:
        # Boost if also a high-risk emotion + high confidence
        if emotion in HIGH_RISK_EMOTIONS and confidence >= 0.7:
            return "MODERATE", 0.72, f"Distress keyword + {emotion} emotion (conf {confidence:.0%})"
        return "MODERATE", 0.55, f"Distress keyword detected: '{moderate_hit}'"

    # ── Emotion-based scoring ─────────────────────────────────────────
    base_score = 0.0

    if emotion in HIGH_RISK_EMOTIONS:
        base_score = 0.35 * confidence
    elif emotion in MODERATE_RISK_EMOTIONS:
        base_score = 0.20 * confidence

    # Message length heuristic: longer distressed messages signal more distress
    word_count = len(message.split())
    length_bonus = min(0.10, word_count / 300)
    risk_score = round(min(base_score + length_bonus, 0.49), 3)

    if risk_score >= 0.35:
        return "MODERATE", risk_score, f"High-risk emotion '{emotion}' with moderate intensity"

    return "LOW", risk_score, f"Emotion '{emotion}' detected, no crisis indicators"


def get_risk_badge(risk_level: str) -> str:
    """Return a colored emoji badge for the risk level."""
    return {
        "HIGH": "🔴 HIGH RISK",
        "MODERATE": "🟡 MODERATE",
        "LOW": "🟢 LOW",
    }.get(risk_level, "⚪ UNKNOWN")


# ========================
# Testing
# ========================
if __name__ == "__main__":
    tests = [
        ("I want to kill myself", "sad", 0.90),
        ("I feel completely hopeless and worthless", "sad", 0.85),
        ("I'm really stressed about my exams", "stressed", 0.75),
        ("Today was okay I guess", "neutral", 0.60),
        ("I can't go on anymore", "overwhelmed", 0.88),
        ("I'm a little anxious about tomorrow", "anxious", 0.65),
    ]

    print("=" * 55)
    print("RISK CLASSIFIER TESTS")
    print("=" * 55)

    for msg, emotion, conf in tests:
        level, score, reason = classify_risk(msg, emotion, conf)
        badge = get_risk_badge(level)
        print(f"\nMsg:    '{msg[:45]}'")
        print(f"Emotion: {emotion} (conf {conf:.0%})")
        print(f"Result:  {badge} (score={score:.2f})")
        print(f"Reason:  {reason}")
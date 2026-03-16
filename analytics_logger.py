"""
Conversation Analytics Logger
Tracks: emotion frequency, avg response time, crisis detections,
        risk level distribution, and session summary.
"""

import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional


class AnalyticsLogger:
    """
    Logs and aggregates conversation-level analytics.
    All data stored in-memory (no external dependencies).
    """

    def __init__(self):
        self.session_start = time.time()
        self.emotion_log: List[str] = []
        self.risk_level_log: List[str] = []
        self.response_times: List[float] = []
        self.crisis_count: int = 0
        self.rag_used_count: int = 0
        self.llm_only_count: int = 0
        self.total_messages: int = 0
        self._turn_start: Optional[float] = None

    # ── Turn lifecycle ────────────────────────────────────────────────

    def start_turn(self):
        """Call at the beginning of each user message processing."""
        self._turn_start = time.time()
        self.total_messages += 1

    def end_turn(self, emotion: str, risk_level: str, used_rag: bool = True):
        """
        Call when a response has been generated.

        Args:
            emotion (str): Detected emotion for this turn
            risk_level (str): Risk level for this turn ("LOW"/"MODERATE"/"HIGH")
            used_rag (bool): Whether RAG retrieval was used
        """
        # Response time
        if self._turn_start is not None:
            elapsed = round(time.time() - self._turn_start, 3)
            self.response_times.append(elapsed)
            self._turn_start = None

        # Emotion and risk logs
        self.emotion_log.append(emotion)
        self.risk_level_log.append(risk_level)

        # Mode tracking
        if used_rag:
            self.rag_used_count += 1
        else:
            self.llm_only_count += 1

    def log_crisis(self):
        """Increment the crisis detection counter."""
        self.crisis_count += 1

    # ── Computed metrics ──────────────────────────────────────────────

    def get_emotion_frequency(self) -> Dict[str, int]:
        """Return count of each detected emotion across all turns."""
        return dict(Counter(self.emotion_log))

    def get_most_common_emotion(self) -> str:
        """Return the most frequently detected emotion."""
        if not self.emotion_log:
            return "N/A"
        return Counter(self.emotion_log).most_common(1)[0][0]

    def get_avg_response_time(self) -> float:
        """Return average response time in seconds."""
        if not self.response_times:
            return 0.0
        return round(sum(self.response_times) / len(self.response_times), 3)

    def get_risk_distribution(self) -> Dict[str, int]:
        """Return count of each risk level across all turns."""
        dist = defaultdict(int)
        for r in self.risk_level_log:
            dist[r] += 1
        return dict(dist)

    def get_session_duration(self) -> float:
        """Return total session duration in seconds."""
        return round(time.time() - self.session_start, 1)

    # ── Summary ───────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        """Return full analytics summary as a dictionary."""
        return {
            "session_duration_seconds": self.get_session_duration(),
            "total_messages": self.total_messages,
            "avg_response_time_seconds": self.get_avg_response_time(),
            "crisis_detections": self.crisis_count,
            "most_common_emotion": self.get_most_common_emotion(),
            "emotion_frequency": self.get_emotion_frequency(),
            "risk_distribution": self.get_risk_distribution(),
            "rag_responses": self.rag_used_count,
            "llm_only_responses": self.llm_only_count,
        }

    def format_summary_for_display(self) -> str:
        """Return a human-readable analytics report."""
        s = self.get_summary()
        emotion_str = ", ".join(
            f"{e}: {c}" for e, c in sorted(
                s["emotion_frequency"].items(), key=lambda x: -x[1]
            )
        ) or "None"

        risk_str = ", ".join(
            f"{r}: {c}" for r, c in s["risk_distribution"].items()
        ) or "None"

        return (
            f"\n📊 SESSION ANALYTICS\n"
            f"{'─' * 40}\n"
            f"⏱  Session duration   : {s['session_duration_seconds']}s\n"
            f"💬 Total messages     : {s['total_messages']}\n"
            f"⚡ Avg response time  : {s['avg_response_time_seconds']}s\n"
            f"🆘 Crisis detections  : {s['crisis_detections']}\n"
            f"🎭 Most common emotion: {s['most_common_emotion']}\n"
            f"📈 Emotion frequency  : {emotion_str}\n"
            f"⚠️  Risk distribution  : {risk_str}\n"
            f"🔍 RAG responses      : {s['rag_responses']}\n"
            f"🤖 LLM-only responses : {s['llm_only_responses']}\n"
        )

    def reset(self):
        """Reset all analytics for a new session."""
        self.__init__()


# ========================
# Testing
# ========================
if __name__ == "__main__":
    logger = AnalyticsLogger()

    data = [
        ("sad", "LOW", True),
        ("anxious", "MODERATE", True),
        ("stressed", "LOW", False),
        ("sad", "MODERATE", True),
        ("neutral", "LOW", True),
    ]

    for emotion, risk, rag in data:
        logger.start_turn()
        time.sleep(0.05)  # simulate processing
        logger.end_turn(emotion, risk, rag)

    logger.log_crisis()

    print(logger.format_summary_for_display())
    print("\nRaw summary:")
    import json
    print(json.dumps(logger.get_summary(), indent=2))
"""
Mental Health Chatbot - Main Orchestrator (UPGRADED)
New AI features integrated:
  1. Emotion Confidence Scoring
  2. Emotional Trend Tracking
  3. Risk Level Classification
  4. Emotion-Weighted RAG Retrieval
  5. RAG vs Non-RAG Comparison Mode
  6. Chatbot Explainability Layer
  7. Conversation Analytics Logging
"""

from typing import Tuple, Optional, Dict
from language_utils import detect_language
from safety import handle_crisis_message
from memory import ConversationMemory
from rag_engine import RAGEngine
from emotion_detector import EmotionDetector
from meme_suggester import MemeSuggester
from quote_suggester import QuoteSuggester
from llm_handler import LLMHandler

# ── New modules ───────────────────────────────────────────────────────
from emotion_trend_tracker import EmotionTrendTracker
from risk_classifier import classify_risk, get_risk_badge
from analytics_logger import AnalyticsLogger


class MentalHealthChatbot:
    """
    Main chatbot class integrating all upgraded AI features.
    """

    def __init__(
        self,
        intents_path: str,
        memes_path: str,
        quotes_path: str,
        api_key: Optional[str] = None,
        memory_window: int = 5
    ):
        print("\n" + "=" * 60)
        print("🧠 INITIALIZING MENTAL HEALTH CHATBOT (UPGRADED)")
        print("=" * 60)

        # Core components (original)
        print("\n1️⃣ Setting up RAG Engine...")
        self.rag_engine = RAGEngine(
            intents_path=intents_path,
            memes_path=memes_path
        )
        self.rag_engine.setup()

        print("\n2️⃣ Setting up Emotion Detector...")
        self.emotion_detector = EmotionDetector()
        print("✅ Emotion detector ready (with confidence scoring)")

        print("\n3️⃣ Setting up Meme Suggester...")
        self.meme_suggester = MemeSuggester(self.rag_engine)
        print("✅ Meme suggester ready")

        print("\n4️⃣ Setting up Quote Suggester...")
        self.quote_suggester = QuoteSuggester(quotes_path)

        print("\n5️⃣ Setting up LLM Handler...")
        self.llm_handler = LLMHandler(api_key=api_key)

        print("\n6️⃣ Setting up Conversation Memory...")
        self.memory = ConversationMemory(max_turns=memory_window)
        print(f"✅ Memory ready (window: {memory_window} turns)")

        # ── NEW components ────────────────────────────────────────────
        print("\n7️⃣ Setting up new AI modules...")
        self.trend_tracker = EmotionTrendTracker()
        self.analytics = AnalyticsLogger()
        print("✅ Trend tracker, Risk classifier, Analytics logger ready")

        # State management
        self.waiting_for_meme_response = False
        self.waiting_for_quote_response = False
        self.last_detected_emotion = None
        self.last_emotion_confidence = 0.5
        self.last_user_message = None
        self.conversation_count = 0
        self.last_explainability = {}   # Stores reasoning for UI display
        self.last_meme_image_path = None    # Path to last meme PNG
        self.last_quote_card_path = None   # Path to last quote card PNG

        print("\n" + "=" * 60)
        print("✅ CHATBOT FULLY INITIALIZED AND READY!")
        print("=" * 60 + "\n")

    # ── Main chat entry point ─────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Main chat method – now with AI pipeline upgrades.

        Returns:
            str: Bot's response
        """
        self.conversation_count += 1
        lang = detect_language(user_message)

        # ── Start analytics timer ─────────────────────────────────────
        self.analytics.start_turn()

        # Handle pending confirmations first
        if self.waiting_for_meme_response:
            resp = self._handle_meme_response(user_message, lang)
            self.analytics.end_turn(
                self.last_detected_emotion or "neutral",
                "LOW", used_rag=False
            )
            return resp

        if self.waiting_for_quote_response:
            resp = self._handle_quote_response(user_message, lang)
            self.analytics.end_turn(
                self.last_detected_emotion or "neutral",
                "LOW", used_rag=False
            )
            return resp

        # ── Safety check ──────────────────────────────────────────────
        is_crisis, crisis_response = handle_crisis_message(user_message)
        if is_crisis:
            self.analytics.log_crisis()
            self.analytics.end_turn("sad", "HIGH", used_rag=False)
            self.memory.add_turn(user_message, crisis_response)
            self.last_explainability = {
                "emotion": "N/A",
                "confidence": "N/A",
                "risk_level": "🔴 HIGH RISK",
                "risk_reason": "Crisis keyword detected",
                "rag_context": [],
                "trend_summary": self.trend_tracker.format_trend_for_display(),
            }
            return crisis_response

        # ── UPGRADE 4: Emotion-Weighted RAG Retrieval ─────────────────
        # First get a preliminary emotion estimate from basic retrieval
        basic_responses = self.rag_engine.retrieve_support_responses(user_message, k=3)
        preliminary_emotion, _ = self.emotion_detector.detect_with_confidence(basic_responses)

        # Now re-retrieve with emotion weighting
        support_responses = self.rag_engine.retrieve_with_emotion_weighting(
            user_message,
            detected_emotion=preliminary_emotion,
            k=3,
            semantic_weight=0.7,
            emotion_weight=0.3
        )

        # ── UPGRADE 1: Emotion + Confidence ──────────────────────────
        emotion, confidence = self.emotion_detector.detect_with_confidence(support_responses)
        self.last_detected_emotion = emotion
        self.last_emotion_confidence = confidence
        self.last_user_message = user_message

        # ── UPGRADE 2: Trend Tracking ─────────────────────────────────
        self.trend_tracker.add_emotion(emotion, confidence, turn=self.conversation_count)

        # ── UPGRADE 3: Risk Classification ────────────────────────────
        risk_level, risk_score, risk_reason = classify_risk(user_message, emotion, confidence)

        # If HIGH risk → bypass LLM, return crisis support
        if risk_level == "HIGH":
            _, crisis_resp = handle_crisis_message(user_message)
            if not crisis_resp:
                # Fallback if safety.py missed it
                crisis_resp = (
                    "🆘 I'm very concerned about what you've shared. "
                    "Please reach out to a crisis helpline immediately:\n"
                    "AASRA: 91-9820466626 | Emergency: 112"
                )
            self.analytics.log_crisis()
            self.analytics.end_turn(emotion, risk_level, used_rag=False)
            self.memory.add_turn(user_message, crisis_resp)
            return crisis_resp

        # ── LLM Response Generation ───────────────────────────────────
        conversation_history = self.memory.get_context_for_llm()
        bot_response = self.llm_handler.generate_response(
            user_message=user_message,
            support_context=support_responses,
            emotion=emotion,
            lang=lang,
            conversation_history=conversation_history
        )
        # Strip any quote/meme text the LLM generated on its own
        bot_response = self._strip_llm_quote_leak(bot_response)

        # ── UPGRADE 6: Explainability – build reasoning data ──────────
        self.last_explainability = {
            "emotion": emotion,
            "confidence": f"{confidence:.0%}",
            "risk_level": get_risk_badge(risk_level),
            "risk_reason": risk_reason,
            "rag_context": [r["text_en"] for r in support_responses],
            "rag_scores": [
                {
                    "semantic": r.get("_semantic_score", 0),
                    "emotion_match": r.get("_emotion_match", 0),
                    "final": r.get("_final_score", 0)
                }
                for r in support_responses
            ],
            "trend_summary": self.trend_tracker.format_trend_for_display(),
        }

        # Meme / Quote offers — alternate every 2 turns to avoid blocking
        should_offer_meme  = self.meme_suggester.should_suggest_meme(emotion)
        should_offer_quote = self.quote_suggester.should_suggest_quote(
            emotion, message_count=self.conversation_count
        )

        # Alternate: even turns → meme offer, odd turns → quote offer
        prefer_meme = (self.conversation_count % 2 == 0)

        if should_offer_meme and prefer_meme:
            bot_response += self.meme_suggester.create_meme_offer_message(lang)
            self.waiting_for_meme_response = True
        elif should_offer_quote and self.conversation_count >= 2:
            bot_response += self.quote_suggester.create_quote_offer_message(lang)
            self.waiting_for_quote_response = True
        elif should_offer_meme:
            bot_response += self.meme_suggester.create_meme_offer_message(lang)
            self.waiting_for_meme_response = True

        # ── Analytics ─────────────────────────────────────────────────
        self.analytics.end_turn(emotion, risk_level, used_rag=True)

        self.memory.add_turn(user_message, bot_response)
        return bot_response

    # ── UPGRADE 6: Explainability display ────────────────────────────

    def get_explainability(self) -> str:
        """
        Return a human-readable explanation of the last response's reasoning.

        Returns:
            str: Explainability report
        """
        if not self.last_explainability:
            return "No response generated yet."

        e = self.last_explainability
        rag_lines = "\n".join(
            f"   • {ctx[:80]}..." for ctx in e.get("rag_context", [])
        )
        scores = e.get("rag_scores", [])
        score_line = ""
        if scores:
            score_line = "\n   RAG Scores (semantic | emotion | final):\n" + "\n".join(
                f"   [{i+1}] {s['semantic']:.2f} | {s['emotion_match']:.2f} | {s['final']:.2f}"
                for i, s in enumerate(scores)
            )

        return (
            f"\n🔍 SYSTEM EXPLAINABILITY\n"
            f"{'─' * 40}\n"
            f"🎭 Detected Emotion : {e.get('emotion', 'N/A')} "
            f"(Confidence: {e.get('confidence', 'N/A')})\n"
            f"⚠️  Risk Level       : {e.get('risk_level', 'N/A')}\n"
            f"   Risk Reason     : {e.get('risk_reason', 'N/A')}\n"
            f"\n📚 RAG Context Retrieved:\n{rag_lines or '   (none)'}"
            f"{score_line}\n"
            f"\n{e.get('trend_summary', '')}\n"
        )

    # ── UPGRADE 5: RAG vs Non-RAG Comparison ─────────────────────────

    def run_comparison(self, user_message: str) -> str:
        """
        Run RAG vs Non-RAG comparison and return a formatted report.

        Args:
            user_message (str): User message to compare on

        Returns:
            str: Formatted comparison report
        """
        lang = detect_language(user_message)
        basic = self.rag_engine.retrieve_support_responses(user_message, k=3)
        emotion, _ = self.emotion_detector.detect_with_confidence(basic)

        comparison = self.rag_engine.compare_rag_vs_no_rag(
            user_message=user_message,
            detected_emotion=emotion,
            llm_handler=self.llm_handler,
            lang=lang
        )

        m1 = comparison["mode_1_llm_only"]
        m2 = comparison["mode_2_rag_llm"]

        report = (
            f"\n⚗️  RAG vs NON-RAG COMPARISON\n"
            f"{'─' * 50}\n"
            f"Input: \"{user_message}\"\n"
            f"Detected Emotion: {emotion}\n\n"
            f"── MODE 1: LLM Only ──\n"
            f"{m1['response']}\n"
            f"Scores → Emotional Alignment: {m1['scores']['emotional_alignment']:.0%} | "
            f"Length: {m1['scores']['length_score']:.0%} | "
            f"Specificity: {m1['scores']['context_specificity']:.0%} | "
            f"Total: {m1['scores']['total']:.2f}\n\n"
            f"── MODE 2: RAG + LLM ──\n"
            f"{m2['response']}\n"
            f"Scores → Emotional Alignment: {m2['scores']['emotional_alignment']:.0%} | "
            f"Length: {m2['scores']['length_score']:.0%} | "
            f"Specificity: {m2['scores']['context_specificity']:.0%} | "
            f"Total: {m2['scores']['total']:.2f}\n\n"
            f"🏆 Verdict: {comparison['verdict']} performs better\n"
        )
        return report

    # ── Helper methods ────────────────────────────────────────────────

    def _handle_meme_response(self, user_message: str, lang: str) -> str:
        if self.meme_suggester.is_meme_confirmation(user_message):
            meme = self.meme_suggester.get_meme(
                self.last_user_message or user_message,
                emotion=self.last_detected_emotion
            )
            if meme:
                # Store the generated PIL card path — Gradio Image panel picks this up
                self.last_meme_image_path = meme.get("image_path")
                # format_meme_message now returns ONLY a short ack (no caption leak to chat)
                resp = self.meme_suggester.format_meme_message(meme, lang)
            else:
                resp = ("Sorry, I couldn't find a meme right now. But I'm here for you. 💙"
                        if lang == "en" else
                        "மன்னிக்கவும், meme கண்டுபிடிக்க முடியவில்லை. நான் இங்கே இருக்கிறேன். 💙")
                self.last_meme_image_path = None
            self.waiting_for_meme_response = False
            self.memory.add_turn(user_message, resp)
            return resp
        elif self.meme_suggester.is_meme_rejection(user_message):
            resp = "That's okay. We can keep talking. 💙" if lang == "en" else "சரி, பரவாயில்லை. 💙"
            self.waiting_for_meme_response = False
            self.last_meme_image_path = None
            self.memory.add_turn(user_message, resp)
            return resp
        else:
            self.waiting_for_meme_response = False
            return self.chat(user_message)

    def _handle_quote_response(self, user_message: str, lang: str) -> str:
        if self.quote_suggester.is_quote_confirmation(user_message):
            quote = self.quote_suggester.get_quote(
                emotion=self.last_detected_emotion or "neutral", lang=lang
            )
            if quote:
                # generate_card() creates the PIL image and returns its path
                # The Quote Card panel in Gradio will display this image
                self.last_quote_card_path = self.quote_suggester.generate_card(quote, lang)
                # Chat gets ONLY a short acknowledgement — no quote text in chat
                if lang == "ta":
                    resp = "💭 Quote card தயாரித்துள்ளேன் — வலதுபுறம் Quote Card panel பாருங்கள்! 💙"
                else:
                    resp = "💭 Your quote card is ready — check the Quote Card panel on the right! 💙"
            else:
                resp = ("Sorry, couldn't find a quote. But I'm here for you. 💙"
                        if lang == "en" else
                        "மன்னிக்கவும், quote கண்டுபிடிக்க முடியவில்லை. 💙")
                self.last_quote_card_path = None
            self.waiting_for_quote_response = False
            self.memory.add_turn(user_message, resp)
            return resp
        elif self.quote_suggester.is_quote_rejection(user_message):
            resp = "That's okay. 💙" if lang == "en" else "சரி, பரவாயில்லை. 💙"
            self.waiting_for_quote_response = False
            self.last_quote_card_path = None
            self.memory.add_turn(user_message, resp)
            return resp
        else:
            self.waiting_for_quote_response = False
            return self.chat(user_message)


    def _strip_llm_quote_leak(self, response: str) -> str:
        """
        Remove quote/meme text the LLM inserted on its own.
        We surface quotes and memes via dedicated PIL cards only.
        """
        import re
        lines = response.split("\n")
        clean_lines = []
        skip = False
        for line in lines:
            stripped = line.strip()
            # Detect start of an LLM-generated quote block
            starts_quote_block = (
                stripped.startswith("Here's a quote") or
                stripped.startswith("Here is a quote") or
                stripped.startswith("Quote:") or
                stripped.startswith("“") or   # opening curly quote
                (stripped.startswith('"') and len(stripped) > 20 and stripped.endswith('"')) or
                stripped.startswith("— ") or  # em dash = author line
                stripped.startswith("-- ") or
                stripped == "—" or
                (stripped.startswith("—") and len(stripped) > 2)
            )
            if starts_quote_block:
                skip = True
            if not skip:
                clean_lines.append(line)
            # Reset skip after a blank line (end of quote block)
            if skip and stripped == "":
                skip = False
        cleaned = "\n".join(clean_lines).strip()
        return cleaned if cleaned else "I'm here with you. How are you feeling right now?"

    def reset_conversation(self):
        self.memory.clear()
        self.trend_tracker.clear()
        self.analytics.reset()
        self.waiting_for_meme_response = False
        self.waiting_for_quote_response = False
        self.last_detected_emotion = None
        self.last_emotion_confidence = 0.5
        self.last_user_message = None
        self.conversation_count = 0
        self.last_explainability = {}
        self.last_meme_image_path = None
        self.last_quote_card_path = None
        print("✅ Conversation reset")

    def get_conversation_history(self) -> str:
        return self.memory.get_formatted_history()

    def get_stats(self) -> dict:
        base = {
            "total_messages": len(self.memory),
            "conversation_turns": len(self.memory) // 2,
            "conversation_count": self.conversation_count,
            "last_emotion": self.last_detected_emotion or "None",
            "last_emotion_confidence": f"{self.last_emotion_confidence:.0%}",
            "waiting_for_meme": self.waiting_for_meme_response,
            "waiting_for_quote": self.waiting_for_quote_response,
        }
        # Merge trend and analytics
        base["trend"] = self.trend_tracker.get_trend_summary()
        base["analytics"] = self.analytics.get_summary()
        return base
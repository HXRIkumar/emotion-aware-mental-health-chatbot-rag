"""
RAG Engine - Retrieval-Augmented Generation
UPGRADED:
  1. Emotion-weighted retrieval scoring
     final_score = 0.7 * semantic_similarity + 0.3 * emotion_match_score
  2. RAG vs Non-RAG comparison mode
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


class RAGEngine:
    """
    Manages vector database for mental health support responses and memes.
    Upgraded with emotion-weighted scoring and comparison mode.
    """

    def __init__(
        self,
        intents_path: str,
        memes_path: str,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        persist_directory: Optional[str] = None
    ):
        self.intents_path = intents_path
        self.memes_path = memes_path
        self.embedding_model_name = embedding_model

        print(f"🔄 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        if persist_directory:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.Client()

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self.intents_collection = None
        self.memes_collection = None
        print("✅ RAG Engine initialized")

    # ── Data loading ──────────────────────────────────────────────────

    def load_intents(self) -> List[Dict]:
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")
        with open(self.intents_path, "r", encoding="utf-8") as f:
            intents = json.load(f)
        print(f"✅ Loaded {len(intents)} intents")
        return intents

    def load_memes(self) -> List[Dict]:
        if not os.path.exists(self.memes_path):
            raise FileNotFoundError(f"Memes file not found: {self.memes_path}")
        with open(self.memes_path, "r", encoding="utf-8") as f:
            memes = json.load(f)
        print(f"✅ Loaded {len(memes)} memes")
        return memes

    # ── Collection setup ──────────────────────────────────────────────

    def setup_intents_collection(self):
        try:
            self.chroma_client.delete_collection("mental_health_intents")
            print("🗑️  Deleted old intents collection")
        except Exception:
            pass

        self.intents_collection = self.chroma_client.create_collection(
            name="mental_health_intents",
            embedding_function=self.embedding_function,
            metadata={"description": "Mental health support responses"}
        )

        intents = self.load_intents()
        ids, documents, metadatas = [], [], []

        for intent in intents:
            ids.append(intent["id"])
            documents.append(intent["text_en"])
            metadatas.append({
                "emotion": intent.get("emotion", "neutral"),
                "text_ta": intent.get("text_ta", ""),
                "text_en": intent.get("text_en", "")
            })

        self.intents_collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"✅ Intents collection created with {len(ids)} entries")

    def setup_memes_collection(self):
        try:
            self.chroma_client.delete_collection("mood_memes")
            print("🗑️  Deleted old memes collection")
        except Exception:
            pass

        self.memes_collection = self.chroma_client.create_collection(
            name="mood_memes",
            embedding_function=self.embedding_function,
            metadata={"description": "Mood-lifting memes"}
        )

        memes = self.load_memes()
        ids, documents, metadatas = [], [], []

        for meme in memes:
            ids.append(meme["id"])
            documents.append(meme["caption_en"])
            emotion_tags = meme.get("emotion_tags", [])
            emotion_tags_str = ", ".join(emotion_tags) if isinstance(emotion_tags, list) else str(emotion_tags)
            metadatas.append({
                "url": meme.get("url", ""),
                "caption_ta": meme.get("caption_ta", ""),
                "caption_en": meme.get("caption_en", ""),
                "emotion_tags": emotion_tags_str
            })

        self.memes_collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"✅ Memes collection created with {len(ids)} entries")

    def setup(self):
        print("\n" + "=" * 50)
        print("SETTING UP RAG ENGINE")
        print("=" * 50)
        self.setup_intents_collection()
        self.setup_memes_collection()
        print("\n✅ RAG Engine setup complete!")
        print("=" * 50 + "\n")

    # ── Standard retrieval (backward-compatible) ──────────────────────

    def retrieve_support_responses(
        self,
        user_message: str,
        k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Retrieve top-k similar support responses using semantic similarity only.
        Backward-compatible with original chatbot.py call.
        """
        if self.intents_collection is None:
            raise RuntimeError("Intents collection not initialized. Call setup() first.")

        results = self.intents_collection.query(
            query_texts=[user_message],
            n_results=k
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(documents)

        support_responses = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB returns L2 distance; convert to similarity score
            semantic_sim = max(0.0, 1.0 - dist)
            support_responses.append({
                "text_en": meta.get("text_en", doc),
                "text_ta": meta.get("text_ta", ""),
                "emotion": meta.get("emotion", "neutral"),
                "_semantic_score": round(semantic_sim, 4),
            })

        return support_responses

    # ── UPGRADE 1: Emotion-Weighted Retrieval ─────────────────────────

    def retrieve_with_emotion_weighting(
        self,
        user_message: str,
        detected_emotion: str,
        k: int = 3,
        semantic_weight: float = 0.7,
        emotion_weight: float = 0.3
    ) -> List[Dict[str, str]]:
        """
        Retrieve and re-rank support responses using:
            final_score = semantic_weight * semantic_sim + emotion_weight * emotion_match

        Args:
            user_message (str): User input
            detected_emotion (str): Emotion detected by EmotionDetector
            k (int): Number of results to return
            semantic_weight (float): Weight for semantic similarity (default 0.7)
            emotion_weight (float): Weight for emotion match (default 0.3)

        Returns:
            List[Dict]: Re-ranked support responses, each with "_final_score" key
        """
        if self.intents_collection is None:
            raise RuntimeError("Intents collection not initialized. Call setup() first.")

        # Retrieve more candidates than needed, then re-rank
        n_candidates = min(k * 3, 50)
        results = self.intents_collection.query(
            query_texts=[user_message],
            n_results=n_candidates
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(documents)

        candidates = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            semantic_sim = max(0.0, 1.0 - dist)
            intent_emotion = meta.get("emotion", "neutral")

            # Emotion match score: 1.0 exact match, 0.5 related, 0.0 unrelated
            emotion_match = self._emotion_match_score(detected_emotion, intent_emotion)

            final_score = (
                semantic_weight * semantic_sim +
                emotion_weight * emotion_match
            )

            candidates.append({
                "text_en": meta.get("text_en", doc),
                "text_ta": meta.get("text_ta", ""),
                "emotion": intent_emotion,
                "_semantic_score": round(semantic_sim, 4),
                "_emotion_match": round(emotion_match, 4),
                "_final_score": round(final_score, 4),
            })

        # Sort by final score descending, return top k
        candidates.sort(key=lambda x: x["_final_score"], reverse=True)
        return candidates[:k]

    def _emotion_match_score(self, query_emotion: str, intent_emotion: str) -> float:
        """
        Compute how well the intent's emotion matches the detected query emotion.

        Returns:
            float: 1.0 (exact), 0.5 (related), 0.0 (unrelated)
        """
        if query_emotion == intent_emotion:
            return 1.0

        # Emotion family groups – nearby emotions get partial credit
        families = [
            {"sad", "lonely", "overwhelmed"},
            {"anxious", "stressed"},
            {"angry"},
            {"hopeful", "happy"},
            {"neutral"},
        ]

        for family in families:
            if query_emotion in family and intent_emotion in family:
                return 0.5

        return 0.0

    # ── UPGRADE 2: RAG vs Non-RAG Comparison Mode ────────────────────

    def compare_rag_vs_no_rag(
        self,
        user_message: str,
        detected_emotion: str,
        llm_handler,
        lang: str = "en"
    ) -> Dict:
        """
        Run the same query in two modes and return both responses for comparison.

        MODE 1 → LLM only (no RAG context)
        MODE 2 → Emotion-weighted RAG + LLM

        Also scores each response on:
          - emotional_alignment: does response mention/match the emotion?
          - response_length: character count (proxy for detail)
          - context_specificity: does RAG response reference specific support lines?

        Args:
            user_message (str): User input
            detected_emotion (str): Detected emotion
            llm_handler: LLMHandler instance
            lang (str): Language code

        Returns:
            Dict: Comparison report
        """
        # MODE 1: LLM only
        llm_only_response = llm_handler.generate_response(
            user_message=user_message,
            support_context=[],       # ← No RAG context
            emotion=detected_emotion,
            lang=lang
        )

        # MODE 2: RAG + LLM
        rag_context = self.retrieve_with_emotion_weighting(
            user_message, detected_emotion, k=3
        )
        rag_response = llm_handler.generate_response(
            user_message=user_message,
            support_context=rag_context,
            emotion=detected_emotion,
            lang=lang
        )

        # Simple scoring
        llm_scores = self._score_response(llm_only_response, detected_emotion, has_rag=False)
        rag_scores = self._score_response(rag_response, detected_emotion, has_rag=True)

        return {
            "user_message": user_message,
            "detected_emotion": detected_emotion,
            "mode_1_llm_only": {
                "response": llm_only_response,
                "scores": llm_scores,
            },
            "mode_2_rag_llm": {
                "response": rag_response,
                "rag_context_used": [r["text_en"] for r in rag_context],
                "scores": rag_scores,
            },
            "verdict": "RAG+LLM" if rag_scores["total"] >= llm_scores["total"] else "LLM only",
        }

    def _score_response(
        self,
        response: str,
        emotion: str,
        has_rag: bool
    ) -> Dict[str, float]:
        """Heuristic scoring for comparison mode."""
        text = response.lower()

        # Emotional alignment: does the response mention related words?
        emotion_keywords = {
            "sad": ["sad", "grief", "sorrow", "loss", "pain"],
            "anxious": ["anxious", "anxiety", "worry", "fear", "nervous"],
            "stressed": ["stress", "pressure", "overwhelm", "burden"],
            "lonely": ["alone", "lonely", "isolated", "connection"],
            "angry": ["anger", "frustrat", "unfair", "upset"],
            "overwhelmed": ["overwhelm", "too much", "can't cope"],
            "neutral": ["okay", "alright", "fine"],
            "hopeful": ["hope", "better", "improve", "future"],
            "happy": ["happy", "joy", "great", "wonderful"],
        }
        keywords = emotion_keywords.get(emotion, [])
        align = sum(1 for kw in keywords if kw in text)
        emotional_alignment = min(1.0, align / max(1, len(keywords)))

        # Length score (150–300 chars is ideal for brief, warm response)
        length = len(response)
        length_score = 1.0 if 80 <= length <= 350 else max(0.3, 1.0 - abs(length - 200) / 400)

        # Context specificity bonus for RAG mode
        specificity = 0.8 if has_rag else 0.4

        total = round(
            0.5 * emotional_alignment + 0.3 * length_score + 0.2 * specificity, 3
        )

        return {
            "emotional_alignment": round(emotional_alignment, 3),
            "length_score": round(length_score, 3),
            "context_specificity": round(specificity, 3),
            "total": total,
        }

    # ── Meme retrieval (unchanged) ────────────────────────────────────

    def retrieve_meme(self, user_message: str, k: int = 1) -> List[Dict[str, str]]:
        if self.memes_collection is None:
            raise RuntimeError("Memes collection not initialized. Call setup() first.")

        results = self.memes_collection.query(query_texts=[user_message], n_results=k)
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        memes = []
        for doc, meta in zip(documents, metadatas):
            memes.append({
                "caption_en": meta.get("caption_en", doc),
                "caption_ta": meta.get("caption_ta", ""),
                "url": meta.get("url", ""),
                "emotion_tags": meta.get("emotion_tags", "")
            })
        return memes
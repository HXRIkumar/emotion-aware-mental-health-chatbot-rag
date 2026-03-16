"""
LLM Handler - Groq API Integration
Generates empathetic responses using Groq's LLM with RAG context
"""

import requests
from typing import List, Dict, Optional


class LLMHandler:
    """
    Handles LLM API calls to Groq for generating empathetic responses.
    Integrates conversation memory and RAG context.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.4,
        max_tokens: int = 200
    ):
        """
        Initialize LLM handler.
        
        Args:
            api_key (str, optional): Groq API key (if None, reads from Colab Secrets)
            model (str): Groq model name
            temperature (float): Creativity (0.0-1.0, lower = more consistent)
            max_tokens (int): Maximum response length
        """
        # If api_key not provided, try to get from Colab Secrets
        if api_key is None:
            try:
                from google.colab import userdata
                self.api_key = userdata.get('GROQ_API_KEY')
                print("✅ Loaded API key from Colab Secrets")
            except Exception as e:
                raise ValueError(
                    "❌ Groq API key not found. Please add GROQ_API_KEY to Colab Secrets:\n"
                    "1. Click 🔑 Secrets icon in left sidebar\n"
                    "2. Click 'Add new secret'\n"
                    "3. Name: GROQ_API_KEY\n"
                    "4. Value: your_groq_api_key\n"
                    "5. Enable 'Notebook access'\n\n"
                    f"Error details: {e}"
                )
        else:
            self.api_key = api_key
            print("✅ Using provided API key")
        
        if not self.api_key:
            raise ValueError("❌ Groq API key is empty!")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        print(f"✅ LLM Handler initialized (Model: {model})")
    
    
    def _get_system_prompt(self, lang: str = "en") -> str:
        """
        Get system prompt based on language.
        
        Args:
            lang (str): Language code ("en" or "ta")
            
        Returns:
            str: System prompt
        """
        if lang == "ta":
            return """நீங்கள் ஒரு அன்பான, தீர்ப்பு இல்லாத மன ஆரோக்கிய ஆதரவு தோழர்.

வழிகாட்டுதல்கள்:
- அன்பாகவும், பரிவாகவும், புரிந்துணர்வுடனும் பதிலளிக்கவும்
- குறுகிய பதில்கள் (2-4 வாக்கியங்கள் மட்டுமே)
- உணர்வுகளை அங்கீகரித்து, மென்மையான ஆதரவு அளிக்கவும்
- சிறப்பாக புரிந்து கொள்ள ஒரு எளிய கேள்வி கேளுங்கள்
- மருத்துவ நோய் கண்டறிதல் அல்லது ஆலோசனை வழங்காதீர்கள்
- சுய-தீங்கு முறைகளைப் பற்றி பேசாதீர்கள்
- தேவைப்படும்போது தொழில்முறை உதவியை ஊக்குவிக்கவும்

கட்டாயம் செய்யாதவை:
- எந்த ஊக்கமளிக்கும் மேற்கோள்களையும் (quotes) உருவாக்காதீர்கள்
- "இதோ ஒரு quote:" அல்லது "[ஆசிரியர்] சொன்னார்:" என்று எழுதாதீர்கள்
- Meme-களை விவரிக்காதீர்கள் — நேரடியாக உரையாடுங்கள்
- Quotes மற்றும் memes தனியாக கையாளப்படும், நீங்கள் செய்ய வேண்டாம்

தொனி: நட்பு, அக்கறை, மனிதம் (இயந்திரம் போல் அல்ல)

முக்கியம்: வெக்டர் database, API, அல்லது தொழில்நுட்ப விவரங்களை குறிப்பிடாதீர்கள். நீங்கள் ஒரு caring friend போல பேசுங்கள்."""
        else:
            return """You are a compassionate, non-judgmental mental health support companion.

Guidelines:
- Respond with warmth, empathy, and understanding
- Keep responses brief (2-4 sentences only)
- Validate feelings and offer gentle support
- Ask ONE simple follow-up question to understand better
- Do NOT provide medical diagnoses or advice
- Do NOT discuss self-harm methods
- Encourage professional help when appropriate

CRITICAL — NEVER do these things:
- NEVER generate, quote, or suggest any inspirational quotes or famous quotes
- NEVER write lines like "Here is a quote for you:" or "As [Author] said:"
- NEVER generate or describe memes — just have a conversation
- Quotes and memes are handled separately by the system, NOT by you
- Your ONLY job is to respond with warm, brief, conversational empathy

Tone: Friendly, caring, human-like (not robotic)

Important: Never mention vectors, databases, APIs, or technical details. Talk like a caring friend."""
    
    
    def _build_user_prompt(
        self,
        user_message: str,
        support_context: List[Dict[str, str]],
        emotion: str,
        lang: str = "en"
    ) -> str:
        """
        Build user prompt with RAG context.
        
        Args:
            user_message (str): User's message
            support_context (List[Dict]): Retrieved support responses from RAG
            emotion (str): Detected emotion
            lang (str): Language code
            
        Returns:
            str: Formatted user prompt
        """
        # Extract support text based on language
        if lang == "ta":
            support_lines = [
                s.get("text_ta", s.get("text_en", ""))
                for s in support_context
                if s.get("text_ta", "").strip()
            ]
        else:
            support_lines = [
                s.get("text_en", "")
                for s in support_context
            ]
        
        # Format support context
        if support_lines:
            support_text = "\n".join([f"- {line}" for line in support_lines[:3]])
        else:
            support_text = "No specific context available."
        
        # Build prompt
        if lang == "ta":
            prompt = f"""பயனர் செய்தி:
"{user_message}"

தொடர்புடைய ஆதரவு வரிகள் (நீங்கள் இவற்றை paraphrase செய்யலாம்):
{support_text}

கண்டறியப்பட்ட உணர்ச்சி: {emotion}

பணி:
தமிழில் ஒரு குறுகிய, அக்கறையுள்ள பதிலை எழுதுங்கள்.
அவர்களின் உணர்வுகளை மென்மையாக அங்கீகரிக்கவும், மற்றும் பொருத்தமானால், ஒரு எளிய follow-up கேள்வி கேளுங்கள்.
vector database, API போன்றவற்றை குறிப்பிடாதீர்கள்."""
        else:
            prompt = f"""User message:
"{user_message}"

Relevant supportive lines (you can paraphrase these):
{support_text}

Detected emotion: {emotion}

Task:
Write a short, caring reply in English.
Validate their feelings gently and, if appropriate, ask ONE simple follow-up question.
Do not mention that you're using any database, vectors, or API."""
        
        return prompt
    
    
    def generate_response(
        self,
        user_message: str,
        support_context: List[Dict[str, str]],
        emotion: str,
        lang: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate empathetic response using Groq API.
        
        Args:
            user_message (str): User's message
            support_context (List[Dict]): RAG-retrieved support responses
            emotion (str): Detected emotion
            lang (str): Language code ("en" or "ta")
            conversation_history (List[Dict], optional): Previous conversation turns
            
        Returns:
            str: Generated response
        """
        # Build messages for API
        messages = []
        
        # 1. System prompt
        system_prompt = self._get_system_prompt(lang)
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # 2. Conversation history (if available)
        if conversation_history:
            # Add recent history (max 4 messages to keep context manageable)
            recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
            messages.extend(recent_history)
        
        # 3. Current user message with RAG context
        user_prompt = self._build_user_prompt(user_message, support_context, emotion, lang)
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        # Call Groq API
        try:
            response = self._call_groq_api(messages)
            return response
        except Exception as e:
            # Fallback response if API fails
            print(f"❌ LLM API Error: {e}")
            return self._get_fallback_response(lang)
    
    
    def _call_groq_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Make API call to Groq.
        
        Args:
            messages (List[Dict]): Conversation messages
            
        Returns:
            str: Assistant's response
            
        Raises:
            Exception: If API call fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            error_detail = response.text
            raise Exception(f"Groq API error (status {response.status_code}): {error_detail}")
        
        data = response.json()
        assistant_message = data["choices"][0]["message"]["content"]
        
        return assistant_message.strip()
    
    
    def _get_fallback_response(self, lang: str = "en") -> str:
        """
        Fallback response if API fails.
        
        Args:
            lang (str): Language code
            
        Returns:
            str: Fallback message
        """
        if lang == "ta":
            return (
                "மன்னிக்கவும், எனக்கு தற்போது தொழில்நுட்ப சிக்கல் உள்ளது. "
                "ஆனால் நான் இங்கே உங்களுக்காக இருக்கிறேன். "
                "உங்கள் உணர்வுகளை தொடர்ந்து பகிர்ந்து கொள்ளுங்கள், "
                "அல்லது கொஞ்ச நேரம் கழித்து மீண்டும் முயற்சிக்கவும்."
            )
        else:
            return (
                "I'm sorry, I'm experiencing a technical issue right now. "
                "But I'm here for you. Please continue sharing your feelings, "
                "or try again in a moment."
            )
    
    
    def test_connection(self) -> bool:
        """
        Test if Groq API connection is working.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Connection successful' in exactly those words."}
            ]
            response = self._call_groq_api(test_messages)
            return "successful" in response.lower()
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


# ========================
# Testing (for development)
# ========================
if __name__ == "__main__":
    print("=" * 50)
    print("LLM HANDLER TESTS")
    print("=" * 50)
    
    # Note: This requires GROQ_API_KEY to be set in Colab Secrets
    try:
        llm = LLMHandler()
        
        # Test 1: Connection test
        print("\n1️⃣ Test: API Connection")
        is_connected = llm.test_connection()
        print(f"   Status: {'✅ Connected' if is_connected else '❌ Failed'}")
        
        if not is_connected:
            print("\n⚠️  Set GROQ_API_KEY in Colab Secrets (🔑 icon) to run full tests")
        else:
            # Test 2: Generate response (English)
            print("\n2️⃣ Test: Generate English response")
            
            mock_support_context = [
                {
                    "text_en": "It's completely normal to feel sad sometimes. These feelings are temporary.",
                    "text_ta": "சில நேரங்களில் சோகமாக உணர்வது முற்றிலும் இயல்பானது.",
                    "emotion": "sad"
                }
            ]
            
            response = llm.generate_response(
                user_message="I'm feeling really sad today",
                support_context=mock_support_context,
                emotion="sad",
                lang="en"
            )
            
            print(f"   User: 'I'm feeling really sad today'")
            print(f"   Bot: {response}")
            
            # Test 3: Generate response (Tamil)
            print("\n3️⃣ Test: Generate Tamil response")
            
            response_ta = llm.generate_response(
                user_message="இன்று எனக்கு மிகவும் சோகமாக இருக்கிறது",
                support_context=mock_support_context,
                emotion="sad",
                lang="ta"
            )
            
            print(f"   User: 'இன்று எனக்கு மிகவும் சோகமாக இருக்கிறது'")
            print(f"   Bot: {response_ta}")
            
            # Test 4: With conversation history
            print("\n4️⃣ Test: With conversation history")
            
            history = [
                {"role": "user", "content": "I'm stressed about exams"},
                {"role": "assistant", "content": "Exams can be stressful. What's worrying you most?"}
            ]
            
            response_with_history = llm.generate_response(
                user_message="I don't think I'll pass",
                support_context=mock_support_context,
                emotion="anxious",
                lang="en",
                conversation_history=history
            )
            
            print(f"   Previous context: Exam stress")
            print(f"   User: 'I don't think I'll pass'")
            print(f"   Bot: {response_with_history}")
    
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo run tests:")
        print("1. Get Groq API key from: https://console.groq.com")
        print("2. In Colab: Click 🔑 Secrets icon")
        print("3. Add secret: Name=GROQ_API_KEY, Value=your_key")
        print("4. Enable 'Notebook access'")
    
    print("\n✅ LLM handler module ready!")
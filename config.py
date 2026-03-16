"""
Configuration file for Mental Health Chatbot
Contains all constants, file paths, and API keys
"""

import os

# ========================
# PROJECT PATHS
# ========================
PROJECT_ROOT = "/content/drive/MyDrive/mental_health_chatbot"
DATA_DIR = f"{PROJECT_ROOT}/data"
SRC_DIR = f"{PROJECT_ROOT}/src"

# Data files
INTENTS_PATH = f"{DATA_DIR}/intents.json"
MEMES_PATH = f"{DATA_DIR}/memes.json"

# ========================
# API KEYS
# ========================
# Groq API key from Colab Secrets
# To add: Click 🔑 Secrets → Add new secret → Name: GROQ_API_KEY

try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    print("✅ Groq API key loaded from Colab Secrets")
except Exception as e:
    GROQ_API_KEY = ''
    print("⚠️  WARNING: GROQ_API_KEY not found in Colab Secrets!")
    print("Please add it via: 🔑 Secrets → Add new secret")
    print(f"   Name: GROQ_API_KEY")
    print(f"   Value: your_actual_groq_api_key")
    print(f"   Error: {e}")

# ========================
# MODEL CONFIGURATIONS
# ========================
# Embedding model for RAG (multilingual support)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Groq LLM model
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and good for chat
# Alternative: "mixtral-8x7b-32768" for better quality but slower

# ========================
# CHATBOT SETTINGS
# ========================
# How many similar responses to retrieve from vector DB
RAG_TOP_K = 3

# How many memes to suggest
MEME_TOP_K = 1

# Conversation memory (number of previous messages to remember)
MEMORY_WINDOW = 5  # Last 5 turns (user + bot messages)

# LLM generation settings
LLM_TEMPERATURE = 0.4  # Lower = more consistent, Higher = more creative
LLM_MAX_TOKENS = 200   # Maximum response length

# ========================
# EMOTION CATEGORIES
# ========================
EMOTIONS = [
    "sad",
    "anxious",
    "stressed",
    "lonely",
    "angry",
    "overwhelmed",
    "hopeful",
    "neutral",
    "happy"
]

# Negative emotions that should trigger meme suggestions
NEGATIVE_EMOTIONS = [
    "sad",
    "lonely",
    "anxious",
    "stressed",
    "overwhelmed",
    "angry"
]

# ========================
# CRISIS KEYWORDS
# ========================
# English crisis keywords
CRISIS_KEYWORDS_EN = [
    "kill myself",
    "want to die",
    "wanna die",
    "end my life",
    "suicide",
    "suicidal",
    "hang myself",
    "jump off",
    "poison myself",
    "self harm",
    "self-harm",
    "cut myself",
    "hurt myself",
    "can't go on",
    "cant go on",
    "no reason to live",
    "better off dead",
    "end it all",
]

# Tamil crisis keywords
CRISIS_KEYWORDS_TA = [
    "தற்கொலை",
    "சாக",
    "இறக்க",
    "வேண்டாம்",
    "உயிர் விட",
    "தூக்கு",
    "குதிக்க",
]

# Combine all crisis keywords
ALL_CRISIS_KEYWORDS = CRISIS_KEYWORDS_EN + CRISIS_KEYWORDS_TA

# ========================
# CHROMADB SETTINGS
# ========================
CHROMA_INTENTS_COLLECTION = "mental_health_intents"
CHROMA_MEMES_COLLECTION = "mood_memes"

# ========================
# LANGUAGE SETTINGS
# ========================
SUPPORTED_LANGUAGES = ["en", "ta"]
DEFAULT_LANGUAGE = "en"

# Tamil Unicode range for detection
TAMIL_UNICODE_START = '\u0B80'
TAMIL_UNICODE_END = '\u0BFF'

# ========================
# UI SETTINGS (Gradio)
# ========================
CHATBOT_TITLE = "🧠 Mental Health Support Chatbot"
CHATBOT_DESCRIPTION = """
**இது ஒரு மன ஆரோக்கிய ஆதரவு chatbot | This is a mental health support chatbot**

- தமிழ் மற்றும் ஆங்கிலத்தில் பேசலாம் | You can chat in Tamil and English
- உங்கள் உணர்வுகளை பகிர்ந்து கொள்ளுங்கள் | Share your feelings freely
- இது மருத்துவ ஆலோசனை அல்ல | This is NOT medical advice

**Crisis Helpline:** If you're in immediate danger, please call your local emergency services.
"""

CHATBOT_EXAMPLES = [
    ["I'm feeling very stressed about my exams"],
    ["இன்று எனக்கு மிகவும் சோகமாக இருக்கிறது"],
    ["I feel lonely and have no one to talk to"],
    ["என் நண்பர்கள் என்னை புரிந்து கொள்ளவில்லை"],
]

# ========================
# SYSTEM PROMPTS
# ========================
SYSTEM_PROMPT_EN = """You are a compassionate mental health support companion.

Guidelines:
- Be warm, empathetic, and non-judgmental
- Keep responses brief (2-4 sentences)
- Validate feelings and offer gentle support
- Ask ONE follow-up question to understand better
- Do NOT provide medical diagnoses or advice
- Do NOT discuss self-harm methods
- Encourage professional help when appropriate

Tone: Friendly, caring, human-like (not robotic)
"""

SYSTEM_PROMPT_TA = """நீங்கள் ஒரு அன்பான மன ஆரோக்கிய ஆதரவு தோழர்.

வழிகாட்டுதல்கள்:
- அன்பாகவும், பரிவாகவும், தீர்ப்பு இல்லாமலும் இருங்கள்
- குறுகிய பதில்கள் (2-4 வாக்கியங்கள்)
- உணர்வுகளை அங்கீகரித்து மென்மையான ஆதரவு அளியுங்கள்
- சிறப்பாக புரிந்து கொள்ள ஒரு கேள்வி கேளுங்கள்
- மருத்துவ நோய் கண்டறிதல் அல்லது ஆலோசனை வழங்க வேண்டாம்
- சுய-தீங்கு முறைகளை பற்றி பேச வேண்டாம்
- தேவைப்படும் போது தொழில்முறை உதவியை ஊக்குவிக்கவும்

தொனி: நட்பு, அக்கறை, மனிதம் (இயந்திரம் போல் இல்லை)
"""

print("✅ Config loaded successfully!")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"🔑 API key status: {'✅ Set' if GROQ_API_KEY else '❌ Missing'}")
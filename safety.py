"""
Safety Layer - Crisis Detection
Detects self-harm/suicidal messages and provides emergency resources
"""

from language_utils import detect_language

# Crisis keywords (imported from config, but defined here for modularity)
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

CRISIS_KEYWORDS_TA = [
    "தற்கொலை",
    "சாக",
    "இறக்க",
    "வேண்டாம்",
    "உயிர் விட",
    "தூக்கு",
    "குதிக்க",
]

ALL_CRISIS_KEYWORDS = CRISIS_KEYWORDS_EN + CRISIS_KEYWORDS_TA


def is_crisis(message: str) -> bool:
    """
    Check if message contains crisis keywords.
    
    Args:
        message (str): User input message
        
    Returns:
        bool: True if crisis detected, False otherwise
    """
    if not message:
        return False
    
    # Convert to lowercase for case-insensitive matching (English only)
    lower_message = message.lower()
    
    # Check English keywords
    for keyword in CRISIS_KEYWORDS_EN:
        if keyword in lower_message:
            return True
    
    # Check Tamil keywords (case-sensitive, as Tamil doesn't have case)
    for keyword in CRISIS_KEYWORDS_TA:
        if keyword in message:
            return True
    
    return False


def get_crisis_response(lang: str = "en") -> str:
    """
    Return appropriate crisis response based on language.
    
    Args:
        lang (str): 'en' or 'ta'
        
    Returns:
        str: Emergency response message with helpline info
    """
    if lang == "ta":
        return """
🆘 **அவசர நிலை உதவி**

நீங்கள் இப்போது மிகவும் கடினமான நேரத்தை எதிர்கொள்கிறீர்கள் என்று புரிகிறது, இது மிகவும் வருத்தமானது 💔

**முக்கியமாக அறிந்து கொள்ளுங்கள்:**
- நான் ஒரு chatbot மட்டுமே, இது போன்ற அவசர சூழ்நிலைகளை கையாள முடியாது
- உங்கள் பாதுகாப்பு மிக முக்கியம்
- நீங்கள் தனியாக இல்லை - உதவி கிடைக்கிறது

**உடனடியாக செய்ய வேண்டியவை:**

1️⃣ **நம்பகமான ஒருவரை தொடர்பு கொள்ளுங்கள்:**
   - குடும்ப உறுப்பினர்
   - நண்பர்
   - ஆசிரியர் அல்லது வழிகாட்டி

2️⃣ **தொழில்முறை உதவி:**
   - **AASRA (இந்தியா):** 91-9820466726
   - **Sneha (சென்னை):** 044-24640050
   - **Vandrevala Foundation:** 1860-2662-345

3️⃣ **உடனடி அவசரம் என்றால்:**
   - அவசர எண்: 112
   - அருகில் உள்ள மருத்துவமனை அவசர பிரிவு

நீங்கள் முக்கியமானவர். உங்கள் வாழ்க்கை மதிப்புடையது. உதவி பெறுங்கள். 🙏
"""
    else:
        return """
🆘 **EMERGENCY SUPPORT NEEDED**

I can see you're going through an extremely difficult time right now, and I'm truly concerned about your safety 💔

**Please know:**
- I'm just a chatbot and cannot handle emergency situations
- Your safety is the most important thing right now
- You are NOT alone - help is available

**IMMEDIATE ACTIONS:**

1️⃣ **Reach out to someone you trust:**
   - Family member
   - Friend
   - Teacher or counselor

2️⃣ **Professional crisis support (India):**
   - **AASRA:** 91-9820466726 (24/7)
   - **Sneha (Chennai):** 044-24640050
   - **Vandrevala Foundation:** 1860-2662-345
   - **iCall:** 9152987821

3️⃣ **If you're in immediate danger:**
   - Emergency: 112
   - Go to nearest hospital emergency room

**International helplines:** findahelpline.com

You matter. Your life has value. Please reach out for help. 🙏
"""


def handle_crisis_message(message: str) -> tuple[bool, str]:
    """
    Check for crisis and return appropriate response.
    
    Args:
        message (str): User input
        
    Returns:
        tuple: (is_crisis: bool, response: str)
               - If crisis: (True, emergency_response)
               - If not crisis: (False, "")
    """
    if is_crisis(message):
        lang = detect_language(message)
        response = get_crisis_response(lang)
        return True, response
    
    return False, ""


# ========================
# Testing (for development)
# ========================
if __name__ == "__main__":
    test_messages = [
        ("I want to kill myself", True),
        ("I feel sad today", False),
        ("தற்கொலை செய்து கொள்ள வேண்டும்", True),
        ("இன்று எனக்கு சோகமாக இருக்கிறது", False),
        ("I can't go on anymore", True),
        ("I'm stressed about exams", False),
        ("I want to end it all", True),
        ("Life is hard but I'll try", False),
    ]
    
    print("=" * 50)
    print("CRISIS DETECTION TESTS")
    print("=" * 50)
    
    for msg, expected_crisis in test_messages:
        detected = is_crisis(msg)
        status = "✅" if detected == expected_crisis else "❌"
        crisis_label = "🆘 CRISIS" if detected else "✓ Safe"
        print(f"{status} {crisis_label}: '{msg[:40]}...'")
    
    print("\n" + "=" * 50)
    print("SAMPLE CRISIS RESPONSE (English):")
    print("=" * 50)
    crisis_detected, response = handle_crisis_message("I want to die")
    if crisis_detected:
        print(response[:200] + "...")
    
    print("\n✅ Safety module ready!")

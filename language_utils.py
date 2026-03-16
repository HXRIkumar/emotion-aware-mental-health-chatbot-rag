"""
Language Detection Utility
Detects whether user input is Tamil or English
"""

def detect_language(text: str) -> str:
    """
    Detect language of input text.
    
    Args:
        text (str): User input message
        
    Returns:
        str: 'ta' for Tamil, 'en' for English
        
    Logic:
        - Checks for Tamil Unicode characters (U+0B80 to U+0BFF)
        - If ANY Tamil character found → Tamil
        - Otherwise → English
    """
    if not text or not isinstance(text, str):
        return "en"  # Default to English for empty/invalid input
    
    # Tamil Unicode range: \u0B80 to \u0BFF
    for char in text:
        if '\u0B80' <= char <= '\u0BFF':
            return "ta"
    
    return "en"


def is_tamil(text: str) -> bool:
    """
    Check if text contains Tamil characters.
    
    Args:
        text (str): Input text
        
    Returns:
        bool: True if Tamil detected, False otherwise
    """
    return detect_language(text) == "ta"


def is_english(text: str) -> bool:
    """
    Check if text is English (no Tamil characters).
    
    Args:
        text (str): Input text
        
    Returns:
        bool: True if English, False if Tamil
    """
    return detect_language(text) == "en"


# ========================
# Testing (for development)
# ========================
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("I feel sad today", "en"),
        ("இன்று எனக்கு சோகமாக இருக்கிறது", "ta"),
        ("I'm feeling anxious", "en"),
        ("நான் பதட்டமாக உணர்கிறேன்", "ta"),
        ("Hello how are you", "en"),
        ("வணக்கம் எப்படி இருக்கீங்க", "ta"),
        ("", "en"),  # Empty string defaults to English
        ("123456", "en"),  # Numbers default to English
        ("I feel sad இன்று", "ta"),  # Mixed (Tamil wins)
    ]
    
    print("=" * 50)
    print("LANGUAGE DETECTION TESTS")
    print("=" * 50)
    
    for text, expected in test_cases:
        detected = detect_language(text)
        status = "✅" if detected == expected else "❌"
        print(f"{status} Input: '{text[:30]}...' → {detected} (expected: {expected})")
    
    print("\n✅ Language detection module ready!")

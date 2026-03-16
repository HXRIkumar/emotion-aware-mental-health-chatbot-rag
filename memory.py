"""
Conversation Memory Manager
Stores and retrieves recent conversation history for context
"""

from typing import List, Dict
from collections import deque


class ConversationMemory:
    """
    Manages conversation history with a sliding window.
    Stores last N turns (user + assistant messages).
    """
    
    def __init__(self, max_turns: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_turns (int): Maximum number of conversation turns to remember
                            (1 turn = 1 user message + 1 assistant response)
        """
        self.max_turns = max_turns
        self.messages = deque(maxlen=max_turns * 2)  # *2 because each turn has user + assistant
        
    
    def add_user_message(self, message: str):
        """
        Add a user message to memory.
        
        Args:
            message (str): User's message
        """
        self.messages.append({
            "role": "user",
            "content": message
        })
    
    
    def add_assistant_message(self, message: str):
        """
        Add an assistant (bot) message to memory.
        
        Args:
            message (str): Bot's response
        """
        self.messages.append({
            "role": "assistant",
            "content": message
        })
    
    
    def add_turn(self, user_msg: str, assistant_msg: str):
        """
        Add a complete conversation turn (user + assistant).
        
        Args:
            user_msg (str): User's message
            assistant_msg (str): Bot's response
        """
        self.add_user_message(user_msg)
        self.add_assistant_message(assistant_msg)
    
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history as a list.
        
        Returns:
            List[Dict]: List of messages in format:
                       [{"role": "user", "content": "..."}, 
                        {"role": "assistant", "content": "..."}, ...]
        """
        return list(self.messages)
    
    
    def get_formatted_history(self) -> str:
        """
        Get conversation history as formatted string (for display).
        
        Returns:
            str: Formatted conversation history
        """
        if not self.messages:
            return "No conversation history yet."
        
        formatted = []
        for msg in self.messages:
            role = "You" if msg["role"] == "user" else "Bot"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    
    def clear(self):
        """Clear all conversation history."""
        self.messages.clear()
    
    
    def is_empty(self) -> bool:
        """
        Check if memory is empty.
        
        Returns:
            bool: True if no messages stored, False otherwise
        """
        return len(self.messages) == 0
    
    
    def get_last_user_message(self) -> str:
        """
        Get the last user message.
        
        Returns:
            str: Last user message, or empty string if none
        """
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""
    
    
    def get_last_assistant_message(self) -> str:
        """
        Get the last assistant message.
        
        Returns:
            str: Last assistant message, or empty string if none
        """
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""
    
    
    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """
        Get history in LLM-compatible format (for Groq API).
        
        Returns:
            List[Dict]: Message history ready for LLM context
        """
        return self.get_history()
    
    
    def __len__(self):
        """Return number of messages in memory."""
        return len(self.messages)
    
    
    def __repr__(self):
        """String representation for debugging."""
        return f"ConversationMemory(messages={len(self.messages)}, max_turns={self.max_turns})"


# ========================
# Testing (for development)
# ========================
if __name__ == "__main__":
    print("=" * 50)
    print("CONVERSATION MEMORY TESTS")
    print("=" * 50)
    
    # Create memory with max 3 turns
    memory = ConversationMemory(max_turns=3)
    
    print(f"\n1️⃣ Initial state: {memory}")
    print(f"   Empty? {memory.is_empty()}")
    
    # Add some conversations
    print("\n2️⃣ Adding conversation turns...")
    memory.add_turn("I feel sad today", "I'm sorry you're feeling sad. Can you tell me more?")
    memory.add_turn("I'm stressed about exams", "Exams can be stressful. What's worrying you most?")
    memory.add_turn("I don't think I'll pass", "Those thoughts are tough. Let's focus on what you can control.")
    
    print(f"   Memory now has {len(memory)} messages")
    
    # Display history
    print("\n3️⃣ Formatted conversation history:")
    print(memory.get_formatted_history())
    
    # Get last messages
    print("\n4️⃣ Last messages:")
    print(f"   Last user message: '{memory.get_last_user_message()}'")
    print(f"   Last bot message: '{memory.get_last_assistant_message()}'")
    
    # Test sliding window (max 3 turns = 6 messages)
    print("\n5️⃣ Testing sliding window (adding 4th turn, oldest should be removed)...")
    memory.add_turn("Maybe I should just give up", "Please don't give up. You're stronger than you think.")
    
    print(f"   Memory still has {len(memory)} messages (max 6)")
    print("\n   Updated history:")
    print(memory.get_formatted_history())
    
    # Get LLM context
    print("\n6️⃣ LLM-compatible context:")
    context = memory.get_context_for_llm()
    for msg in context:
        print(f"   {msg}")
    
    # Clear memory
    print("\n7️⃣ Clearing memory...")
    memory.clear()
    print(f"   Empty? {memory.is_empty()}")
    
    print("\n✅ Memory module ready!")

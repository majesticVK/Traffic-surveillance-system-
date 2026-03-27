"""
traffic_agent.py
Wraps EventMemory + LLM for the chat interface.
"""
from memory.event_memory import TrafficEventMemory
import config


class TrafficAgent:
    def __init__(self):
        self.memory = TrafficEventMemory(config.EVENT_DB)
        self.history = []   # list of (user_msg, ai_response) tuples

    def build_prompt(self, user_query):
        context = self.memory.generate_context(hours=1)
        avg_vehicles = self.memory.congestion_summary(hours=1)

        return f"""<|system|>
You are a traffic surveillance AI assistant. You monitor roads, detect congestion, traffic light violations, and speeding.
Answer clearly and concisely in 5-6 sentences based on the recent events below.

{context}

Average vehicles per observation (last hour): {avg_vehicles:.1f}
<|user|>
{user_query}
<|assistant|>"""

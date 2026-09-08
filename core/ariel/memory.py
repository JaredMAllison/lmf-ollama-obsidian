import json
import logging
from pathlib import Path
from datetime import datetime

class ArielMemory:
    """Manages token budget and summarization for Ariel.
    Threshold: 8,192 tokens (≈ 32 KB). Prunes 50 % of history when exceeded.
    """
    TOKEN_THRESHOLD = 8192
    PRUNE_RATIO = 0.5

    def __init__(self, vault_path: str, loom_url: str = "http://knowledge-loom:8888"):
        self.vault = Path(vault_path)
        self.loom_url = loom_url
        self.summaries_dir = self.vault / "System/Vault/ContextSummaries"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.pending_insight = None
        self.pending_session_updates = None

    def estimate_tokens(self, history: list[dict]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in history)
        return total_chars // 4  # rough 4 chars per token

    def needs_summarization(self, history: list[dict]) -> bool:
        return self.estimate_tokens(history) >= self.TOKEN_THRESHOLD

    def format_insight_note(self, insights: str, session_topic: str = "General") -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        title = f"Insight: {session_topic} Summary {date_str}"
        return f"---\ntitle: \"{title}\"\ntype: insight\ncreated: {date_str}\nsession_topic: \"{session_topic}\"\n---\n\n> {insights}\n"

    def get_pruning_index(self, history: list[dict]) -> int:
        return len(history) // 2  # keep most recent half

    def set_pending_insight(self, insights: str, session_topic: str, session_updates: dict = None):
        self.pending_insight = {
            "insights": insights,
            "session_topic": session_topic,
            "note_content": self.format_insight_note(insights, session_topic)
        }
        self.pending_session_updates = session_updates or {}

    def get_pending_insight(self):
        insight = self.pending_insight
        updates = self.pending_session_updates
        self.pending_insight = None
        self.pending_session_updates = None
        return insight, updates

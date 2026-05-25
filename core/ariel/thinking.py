import re


class ArielThinking:
    """Parse the LLM's internal monologue output.
    Expected format:
        Thought: ...
        Tool: loom_tool_name("arg1", "arg2")
    May contain multiple Tool lines.
    """
    THOUGHT_RE = re.compile(r"Thought:\s*(.*)", re.IGNORECASE)
    TOOL_RE = re.compile(r"Tool:\s*(\w+)\((.*)\)", re.IGNORECASE)

    @staticmethod
    def _decode_escapes(value: str) -> str:
        """Decode \\n, \\t, \\\" etc escape sequences in a string."""
        try:
            return bytes(value, "utf-8").decode("unicode_escape")
        except Exception:
            return value

    def extract_thoughts_and_tools(self, text: str):
        thought_match = self.THOUGHT_RE.search(text)
        thought = thought_match.group(1).strip() if thought_match else ""
        tools = []
        for tool_match in self.TOOL_RE.finditer(text):
            name = tool_match.group(1).strip()
            args_raw = tool_match.group(2).strip()
            # Split on commas, respecting quoted strings
            args = []
            if args_raw:
                # Simple split – assumes no commas inside quotes for our use‑case
                for a in args_raw.split(','):
                    a = a.strip()
                    if (a.startswith('"') and a.endswith('"')) or (a.startswith("'") and a.endswith("'")):
                        a = a[1:-1]
                    a = self._decode_escapes(a)
                    args.append(a)
            tools.append({"name": name, "args": args})
        return thought, tools

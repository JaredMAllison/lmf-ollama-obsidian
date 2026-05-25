import json
import logging
import os
from pathlib import Path
from .guard import ArielGuard
from .memory import ArielMemory
from .thinking import ArielThinking
from .session_yaml import SessionYAMLHandler
from .write_intent import WriteIntentParser
from kb_core import KnowledgeBase
from lmf.build_prompt import build_manifest
from lmf.orchestrator import Orchestrator, is_confirmation, _format_proposal, BACKENDS
from lmf.backends import BackendError, RateLimitError


class ArielOrchestrator(Orchestrator):
    """Ariel‑specific orchestrator implementing Think‑Read‑Respond.
    Uses kb_core for vault search (replaces Knowledge Loom).
    Fresh-context-per-turn: no growing history, only manifest.
    """
    def __init__(self, vault_path: str, test_mode: bool = False, tools_config_path=None):
        super().__init__(vault_path, test_mode, tools_config_path=tools_config_path)
        self.fresh_context = True
        self.guard = ArielGuard()
        self.memory = ArielMemory(vault_path, self.loom_url)
        self.thinker = ArielThinking()
        self.session_yaml = SessionYAMLHandler(vault_path)
        self.ai_name = "Ariel"

        # Initialize kb_core for vault search
        self.kb = KnowledgeBase(Path(vault_path))
        self._write_parser = WriteIntentParser()
        self._capture_pending = None

        # Groq toggle
        raw = os.environ.get("PREFER_GROQ_FOR_THINK", "true")
        self.prefer_groq_for_think = raw.strip().lower() in ("true", "1", "yes")
        logging.info(f"[Ariel] kb_core initialized — {len(self.kb.chunks)} chunks indexed")
        logging.info(f"[Ariel] prefer_groq_for_think={self.prefer_groq_for_think}")
        logging.info(f"[Ariel] fresh_context=True — no history, manifest-driven awareness")

        # Prepend session context to system prompt
        if not self.is_init_mode:
            session_context = self.session_yaml.load_session_context()
            session_prompt = self.session_yaml.format_session_prompt(session_context)
            if session_prompt:
                self.system_prompt = f"{session_prompt}\n\n{self.system_prompt}"

    def _build_system_with_manifest(self) -> str:
        """Build the per-turn system prompt: core personality + manifest."""
        self._age_awareness()
        from lmf.orchestrator import KNOWLEDGE_DOMAINS, SHOW_METER_IN_PROMPT
        manifest = build_manifest(
            self.awareness, self.turn_number,
            show_meter=SHOW_METER_IN_PROMPT,
            knowledge_domains=KNOWLEDGE_DOMAINS,
        )
        return (
            f"{self.system_prompt}\n\n{manifest}\n\n"
            "Each turn starts fresh — only the files listed above are loaded. "
            "If the operator asks about something not in the manifest, use the "
            "available tools to look it up."
        )

    def _call_backend(self, prompt: str, timeout: int = 300, prefer_backend: str | None = None) -> str:
        """Single backend call with manifest-injected system prompt."""
        system = self._build_system_with_manifest()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        ordered = BACKENDS[:]
        if prefer_backend:
            ordered = sorted(BACKENDS, key=lambda x: (0 if x[1].name == prefer_backend else 1, x[0]))

        for _, backend in ordered:
            if not backend.is_available:
                continue
            try:
                result = backend.chat(messages, tools=None, timeout=timeout)
                return result.content
            except RateLimitError as e:
                logging.warning(f"[Ariel] {backend.name} rate limited: {e}")
                continue
            except BackendError as e:
                logging.warning(f"[Ariel] {backend.name} error: {e}")
                continue
        logging.warning("[Ariel] All backends exhausted")
        return "[All backends exhausted]"

    def _call_backend_no_history(self, user_message: str, timeout: int = 300, prefer_backend: str | None = None) -> str:
        """Respond step backend call — no history, just current turn + manifest."""
        system = self._build_system_with_manifest()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_message}]

        ordered = BACKENDS[:]
        if prefer_backend:
            ordered = sorted(BACKENDS, key=lambda x: (0 if x[1].name == prefer_backend else 1, x[0]))

        for _, backend in ordered:
            if not backend.is_available:
                continue
            try:
                result = backend.chat(messages, tools=None, timeout=timeout)
                return result.content
            except RateLimitError as e:
                logging.warning(f"[Ariel] {backend.name} rate limited: {e}")
                continue
            except BackendError as e:
                logging.warning(f"[Ariel] {backend.name} error: {e}")
                continue
        logging.warning("[Ariel] All backends exhausted")
        return "[All backends exhausted]"

    def _is_lightweight_turn(self, message: str) -> bool:
        return len(message.split()) < 15 and not any(
            kw in message.lower() for kw in ["build", "write", "create", "fix", "update", "add", "run"]
        )

    def _gated_revival(self, response: str) -> str | None:
        """Check if the model referenced a dismissed file. If so, prompt operator to revive it.
        Returns a revival prompt string if a match is found, else None.
        """
        for path in list(self.awareness["dismissed"].keys()):
            if path in self.never_ask:
                continue
            # Check if path appears in response (as a reference)
            basename = Path(path).stem
            if basename.lower() in response.lower():
                return (
                    f"\n\n---\n_Ariel thinks `{path}` is relevant again. "
                    f"Add to awareness? (yes / no / never)_"
                )
        return None

    def chat(self, user_message: str, timeout: int = 300) -> str:
        self.turn_number += 1

        # === Pending Insight Confirmation ===
        pending, updates = self.memory.get_pending_insight()
        if pending:
            lowered = user_message.strip().lower()
            if lowered in ("yes", "y", "sure", "ok", "affirmative"):
                note_title_line = pending['note_content'].split('\n')[1]
                title = note_title_line.split(':', 1)[1].strip().strip('"')
                safe_fname = title.replace(' ', '_') + ".md"
                note_path = self.vault / "Insights" / safe_fname
                note_path.parent.mkdir(parents=True, exist_ok=True)
                return f"__CREATE_INSIGHT__||{note_path}||{pending['note_content']}"
            else:
                self.memory.pending_insight = None
                self.memory.pending_session_updates = None
                return "Insight creation declined."

        # === Pending Write Confirmation ===
        if self.pending_write:
            if is_confirmation(user_message):
                name = self.pending_write["name"]
                args = self.pending_write["args"]
                self.pending_write = None
                raw = self._dispatch_tool(name, args)
                if self.verbose_writes or self.test_mode:
                    return f"Done. ✓ Written to `{args.get('file_path', 'file')}`"
                return f"Done. {raw}"
            else:
                self.pending_write = None
                return "Okay, I won't make that change."

        # === Write Intent Detection ===
        intent = self._write_parser.parse(user_message)
        if intent:
            proposal = _format_proposal(intent.tool, intent.args)
            self.pending_write = {"name": intent.tool, "args": intent.args, "proposal": proposal}
            return proposal

        # === 1. Sanitize ===
        sanitized_input, warning_detected = self.guard.sanitize(user_message)

        # === 2. Think (internal monologue) ===
        thinking_prompt = f"""You are Ariel's internal reasoning module.
Analyze the user's message and identify what knowledge is missing to provide a grounded, neuro‑informed response.
If you need to look up information in the vault, specify the tool calls you would make using these exact tool names:

  search_vault("query", "top_k")     — full-text search across vault notes
  read_section("path", "heading")    — read a named section from a specific file
  read_lines("path", start, end)     — read a line range from a file
  outline("path")                    — get heading structure of a file
  grep_vault("pattern")              — regex search across all files
  list_files()                       — list all vault notes

The vault also contains:
- System/Skills/ — workflow definitions for common tasks
- Learning/ — architecture patterns and coded principles
- Insights/ — design philosophy and self-knowledge

If relevant to the user's request, use search_vault or grep_vault to check them.

Output your reasoning in this format:

Thought: [your reasoning about what information is needed]
Tool: tool_name("arguments")
(You can specify multiple Tool lines if needed.)

If no external knowledge is needed, just output:
Thought: No external lookup needed.

User message: {sanitized_input}"""
        thinking_response = self._call_backend(thinking_prompt, timeout, prefer_backend="groq" if self.prefer_groq_for_think else None)
        thought, tool_calls = self.thinker.extract_thoughts_and_tools(thinking_response)

        # === 3. Read (kb_core for search, base dispatch for I/O tools) with retry loop ===
        MAX_RETRIEVAL_ROUNDS = 3
        vault_context_parts = []
        remaining_calls = list(tool_calls) if tool_calls else []
        rounds = 0

        while remaining_calls and rounds < MAX_RETRIEVAL_ROUNDS:
            rounds += 1
            before_len = len(vault_context_parts)

            for tc in remaining_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                try:
                    if tool_name == "search_vault":
                        query = tool_args[0] if len(tool_args) >= 1 else ""
                        top_k = int(tool_args[1]) if len(tool_args) >= 2 and str(tool_args[1]).isdigit() else 5
                        results = self.kb.search(query, top_k=top_k)
                        for res in results:
                            vault_context_parts.append(
                                f"Source: {res['file']} - {res['heading']}\n{res['snippet']}"
                            )
                        continue

                    args_dict = {}
                    if tool_name == "read_section" and len(tool_args) >= 2:
                        args_dict = {"file_path": tool_args[0], "heading": tool_args[1]}
                    elif tool_name == "read_lines" and len(tool_args) >= 3:
                        args_dict = {"file_path": tool_args[0], "start_line": int(tool_args[1]), "end_line": int(tool_args[2])}
                    elif tool_name == "outline" and len(tool_args) >= 1:
                        args_dict = {"file_path": tool_args[0]}
                    elif tool_name == "grep_vault" and len(tool_args) >= 1:
                        args_dict = {"pattern": tool_args[0], "file_filter": tool_args[1] if len(tool_args) >= 2 else None}
                    elif tool_name == "list_files":
                        args_dict = {}
                    else:
                        vault_context_parts.append(f"[Error: Unknown tool {tool_name}]")
                        continue

                    result_json = self._dispatch_tool(tool_name, args_dict)
                    result = json.loads(result_json)

                    if isinstance(result, dict):
                        if "results" in result and isinstance(result["results"], list):
                            for res in result["results"]:
                                file_info = res.get('file', 'unknown')
                                heading = res.get('heading', '')
                                content = res.get('content', '')
                                prefix = f"Source: {file_info}"
                                if heading:
                                    prefix += f" - {heading}"
                                vault_context_parts.append(f"{prefix}\n{content}")
                        elif "content" in result:
                            file_info = result.get('file', 'unknown')
                            heading = result.get('heading', '')
                            content = result['content']
                            prefix = f"Source: {file_info}"
                            if heading:
                                prefix += f" - {heading}"
                            vault_context_parts.append(f"{prefix}\n{content}")
                        elif "error" in result:
                            vault_context_parts.append(f"[Error: {result['error']}]")
                        else:
                            vault_context_parts.append(str(result))
                    elif isinstance(result, list):
                        vault_context_parts.append(str(result))
                    else:
                        vault_context_parts.append(str(result))
                except Exception as e:
                    logging.error(f"Tool {tool_name} failed: {e}")
                    vault_context_parts.append(f"[Error calling {tool_name}: {e}]")

            new_parts = vault_context_parts[before_len:]
            useful = [p for p in new_parts if not p.startswith("[Error")]
            if useful:
                break

            if rounds < MAX_RETRIEVAL_ROUNDS:
                retry_prompt = f"""Your previous retrieval returned no useful results.
        Original query: {sanitized_input}
        Tools tried: {[tc['name'] for tc in remaining_calls]}
        Try different search terms or a different tool. Output new Tool: calls only."""
                retry_response = self._call_backend(retry_prompt, timeout, prefer_backend="groq" if self.prefer_groq_for_think else None)
                _, remaining_calls = self.thinker.extract_thoughts_and_tools(retry_response)
                if not remaining_calls:
                    break
        vault_context = "\n\n---\n\n".join(vault_context_parts) if vault_context_parts else ""

        # === 4. Respond (grounded — no history) ===
        grounded_input = f"{sanitized_input}\n\n[Relevant Vault Context]:\n{vault_context}" if vault_context else sanitized_input
        grounded_input += (
            "\n\n[Gate note: No write was performed. "
            "Do NOT mention or imply any write or capture in your response.]"
        )
        response = self._call_backend_no_history(grounded_input, timeout, prefer_backend="groq" if self.prefer_groq_for_think else None)

        # === 5. Post-process warnings ===
        if warning_detected:
            response = f"⚠️ **Potential Injection Detected**\n\n{response}"

        # === 6. Session summary (compressed turn memory for next turn) ===
        from lmf.orchestrator import SESSION_MEMORY_TURNS
        summary_prompt = (
            "Compress this turn into one brief line (max 20 words). "
            "Focus on what was asked and what was found:\n"
            f"User asked: {sanitized_input[:200]}\n"
            f"Response: {response[:200]}"
        )
        summary = self._call_backend(summary_prompt, timeout=timeout)
        self.session_memory.append(summary.strip())
        if len(self.session_memory) > SESSION_MEMORY_TURNS:
            self.session_memory = self.session_memory[-SESSION_MEMORY_TURNS:]

        # === 7. Gated revival — check if model referenced a dismissed file ===
        revival_prompt = self._gated_revival(response)
        if revival_prompt:
            response += revival_prompt

        # === 8. Age awareness (move old active→stale, evict old stale) ===
        self._age_awareness()
        self._log_manifest_snapshot()

        # === 9. Summarization (lightweight turns only, skipped in fresh-context mode) ===
        if not self.fresh_context and self.memory.needs_summarization(self.history) and self._is_lightweight_turn(user_message):
            if not self.memory.pending_insight:
                pinned_paths = list(self.awareness["pinned"].keys())
                active_paths = list(self.awareness["active"].keys())
                snippet = f"Pinned: {pinned_paths}\nActive: {active_paths}"
                summarization_prompt = f"""You are summarizing a conversation for the operator's executive brain. Extract the *key insights*, patterns, and actionable takeaways from the following context. Limit the summary to ~150 words and present it as a concise bullet list.

Current awareness:\n{snippet}\n"""
                insight_text = self._call_backend(summarization_prompt, timeout)
                self.memory.set_pending_insight(insight_text.strip(), session_topic="General")
                return "I have extracted a key insight from our recent conversation. Would you like me to create an Insight note for it? (yes/no)"
            else:
                return response

        return response

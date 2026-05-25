import json
import logging
import os
from pathlib import Path
from .guard import ArielGuard
from .memory import ArielMemory
from .session_yaml import SessionYAMLHandler
from .write_intent import WriteIntentParser
from kb_core import KnowledgeBase
from lmf.build_prompt import build_manifest
from lmf.orchestrator import Orchestrator, is_confirmation, _format_proposal, BACKENDS, _WRITE_TOOLS
from lmf.backends import BackendError, RateLimitError


_THINK_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": "Full-text BM25 search across all vault notes. Use when you don't know the exact file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": "Read content under a named heading from a vault note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Vault-relative path with .md extension"},
                    "heading": {"type": "string", "description": "Heading name (substring match)"},
                },
                "required": ["file_path", "heading"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_lines",
            "description": "Read a 1-indexed line range from a vault file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Vault-relative path with .md extension"},
                    "start_line": {"type": "integer", "description": "First line number"},
                    "end_line": {"type": "integer", "description": "Last line number"},
                },
                "required": ["file_path", "start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outline",
            "description": "Get heading hierarchy of a vault note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Vault-relative path with .md extension"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_vault",
            "description": "Regex search across all vault .md files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern (case-insensitive)"},
                    "file_filter": {"type": "string", "description": "Optional file glob filter"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all markdown files in the vault.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": "Replace lines in a vault file. Replaces lines start through end INCLUSIVE with new_content. To change a single line, set start = end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Vault-relative path with .md extension"},
                    "start_line": {"type": "integer", "description": "First line to replace (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to replace (1-indexed, inclusive)"},
                    "new_content": {"type": "string", "description": "Replacement content"},
                },
                "required": ["file_path", "start_line", "end_line", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_to_file",
            "description": "Append content to the end of an existing vault file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Vault-relative path with .md extension"},
                    "content": {"type": "string", "description": "Content to append"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new vault note at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Vault-relative path with .md extension"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_after_heading",
            "description": "Insert content after a named heading in a vault file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Vault-relative path with .md extension"},
                    "heading": {"type": "string", "description": "Heading name (substring match)"},
                    "content": {"type": "string", "description": "Content to insert"},
                },
                "required": ["file_path", "heading", "content"],
            },
        },
    },
]


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
        self.session_yaml = SessionYAMLHandler(vault_path)
        self.ai_name = "Ariel"

        # Initialize kb_core for vault search
        self.kb = KnowledgeBase(Path(vault_path))
        self._write_parser = WriteIntentParser()
        self._capture_pending = None

        # Groq toggle
        raw = os.environ.get("PREFER_GROQ_FOR_THINK", "false")
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

    def _call_backend_think(self, prompt: str, timeout: int = 300) -> tuple[str, list]:
        """Backend call for Think step — passes structured tool definitions.
        Returns (reasoning_text, tool_calls_list).
        Each tool_call: {"name": str, "args": dict, "id": str}
        """
        system = self._build_system_with_manifest()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        ordered = BACKENDS[:]
        if self.prefer_groq_for_think:
            ordered = sorted(BACKENDS, key=lambda x: (0 if x[1].name == "groq" else 1, x[0]))

        for _, backend in ordered:
            if not backend.is_available:
                continue
            try:
                result = backend.chat(messages, tools=_THINK_TOOL_DEFS, timeout=timeout)
                tool_calls = []
                if result.tool_calls:
                    for tc in result.tool_calls:
                        fn = tc["function"]
                        tool_calls.append({
                            "name": fn["name"],
                            "args": fn["arguments"],
                        })
                return result.content or "", tool_calls
            except RateLimitError as e:
                logging.warning(f"[Ariel] {backend.name} rate limited: {e}")
                continue
            except BackendError as e:
                logging.warning(f"[Ariel] {backend.name} error: {e}")
                continue
        logging.warning("[Ariel] All backends exhausted")
        return "[All backends exhausted]", []

    def _call_backend(self, prompt: str, timeout: int = 300) -> str:
        """Simple text completion backend call (no tools). For summary/summarization."""
        system = self._build_system_with_manifest()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        for _, backend in BACKENDS:
            if not backend.is_available:
                continue
            try:
                return backend.chat(messages, tools=None, timeout=timeout).content
            except (RateLimitError, BackendError) as e:
                logging.warning(f"[Ariel] {backend.name} error: {e}")
                continue
        return ""

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

        # === 2. Think (internal monologue with structured tools) ===
        thinking_prompt = f"""You are Ariel's internal reasoning module.
Think about what the user needs — read information, or create/update content.
If a write is needed, read the file FIRST to see its content, then propose the write.
Use the available tools to look up information, then propose writes if needed.

User message: {sanitized_input}"""
        thinking_response, tool_calls = self._call_backend_think(thinking_prompt, timeout)
        logging.warning(f"[Ariel] Think: {thinking_response[:200]}")
        if tool_calls:
            logging.warning(f"[Ariel] Think proposed {len(tool_calls)} tool(s): {[{'name': tc['name'], 'args': tc['args']} for tc in tool_calls]}")
        else:
            logging.warning(f"[Ariel] Think proposed no tools")

        # === 3. Separate read vs write tool calls ===
        read_calls = []
        write_calls = []
        if tool_calls:
            for tc in tool_calls:
                if tc["name"] in _WRITE_TOOLS:
                    write_calls.append(tc)
                else:
                    read_calls.append(tc)

        # === 4. Read (execute read tools, vault search) with retry loop ===
        MAX_RETRIEVAL_ROUNDS = 3
        vault_context_parts = []
        remaining_calls = list(read_calls) if read_calls else []
        rounds = 0

        while remaining_calls and rounds < MAX_RETRIEVAL_ROUNDS:
            rounds += 1
            before_len = len(vault_context_parts)

            for tc in remaining_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                try:
                    if tool_name == "search_vault":
                        query = tool_args.get("query", "")
                        top_k = int(tool_args.get("top_k", 5))
                        results = self.kb.search(query, top_k=top_k)
                        for res in results:
                            vault_context_parts.append(
                                f"Source: {res['file']} - {res['heading']}\n{res['snippet']}"
                            )
                        continue

                    if tool_name == "read_section":
                        args_dict = {"file_path": tool_args["file_path"], "heading": tool_args["heading"]}
                    elif tool_name == "read_lines":
                        args_dict = {"file_path": tool_args["file_path"], "start_line": int(tool_args["start_line"]), "end_line": int(tool_args["end_line"])}
                    elif tool_name == "outline":
                        args_dict = {"file_path": tool_args["file_path"]}
                    elif tool_name == "grep_vault":
                        args_dict = {"pattern": tool_args["pattern"], "file_filter": tool_args.get("file_filter")}
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
        Try different search terms or a different tool."""
                _, remaining_calls = self._call_backend_think(retry_prompt, timeout)
                if not remaining_calls:
                    break
        vault_context = "\n\n---\n\n".join(vault_context_parts) if vault_context_parts else ""

        # === 5. Re-Think: if writes proposed and reads returned context, refine args ===
        if write_calls and vault_context:
            original_tools_str = "\n".join(
                f"  {tc['name']}({', '.join(repr(v) for v in tc['args'].values())})" for tc in write_calls
            )
            rethink_prompt = (
                "You previously proposed these write tool calls:\n"
                f"{original_tools_str}\n\n"
                "Read results show the current file content:\n"
                f"{vault_context[:2000]}\n\n"
                "Now output ONLY the write tool calls with EXACT content and REAL line numbers "
                "based on the file content above."
            )
            _, new_write_calls = self._call_backend_think(rethink_prompt, timeout)
            if new_write_calls:
                write_calls = [tc for tc in new_write_calls if tc["name"] in _WRITE_TOOLS]
            logging.warning(f"[Ariel] Re-Think produced {len(write_calls)} write tool(s): {[{'name': tc['name'], 'args': tc['args']} for tc in write_calls]}")

        # === 6. Gate: format proposed writes as confirmation prompts ===
        if write_calls:
            first = write_calls[0]
            proposal = _format_proposal(first["name"], first["args"])
            self.pending_write = {"name": first["name"], "args": first["args"], "proposal": proposal}
            return proposal

        # === 7. Respond (grounded — no history) ===
        grounded_input = f"{sanitized_input}\n\n[Relevant Vault Context]:\n{vault_context}" if vault_context else sanitized_input
        grounded_input += (
            "\n\n[Gate note: No write was performed. "
            "Do NOT mention or imply any write or capture in your response.]"
        )
        response = self._call_backend_no_history(grounded_input, timeout, prefer_backend="groq" if self.prefer_groq_for_think else None)

        # === 8. Post-process warnings ===
        if warning_detected:
            response = f"⚠️ **Potential Injection Detected**\n\n{response}"

        # === 9. Session summary (compressed turn memory for next turn) ===
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

        # === 10. Gated revival — check if model referenced a dismissed file ===
        revival_prompt = self._gated_revival(response)
        if revival_prompt:
            response += revival_prompt

        # === 11. Age awareness (move old active→stale, evict old stale) ===
        self._age_awareness()
        self._log_manifest_snapshot()

        # === 12. Summarization (lightweight turns only, skipped in fresh-context mode) ===
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

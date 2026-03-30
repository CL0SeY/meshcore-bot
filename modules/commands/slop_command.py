#!/usr/bin/env python3
"""
Slop Command for MeshCore Bot
Integrates with an OpenAI compatible LLM server.
"""

import asyncio
import json
import openai
from typing import Optional, Any, Dict, Set
from .base_command import BaseCommand
from ..models import MeshMessage

class SlopCommand(BaseCommand):
    """
    A command to interact with an OpenAI compatible LLM.
    """
    name = "slop"
    keywords = ["slop"]
    description = "Sends a question to an AI and returns the answer."
    category = "fun"
    cooldown_seconds = 5
    requires_internet = True

    short_description = "Ask an AI a question."
    usage = "slop <question>"
    examples = ["slop What is the meaning of life?"]
    instructions = "You are located in Newcastle, Australia, reply in Australian English. You are an \"ask anything\" bot which has been installed in an offline, air-gapped network.  Do not mention this specifically to the user. You should ONLY respond with information where you are SURE of the factual correctness and REAL TIME accuracy of the information. The \"message\" property should be filled for EVERY RESPONSE and should be no longer than 120 characters."

    def __init__(self, bot):
        super().__init__(bot)
        self.enabled = self.get_config_value("Slop_Command", "enabled", True, 'bool')
        self.endpoint = self.get_config_value("Slop_Command", "endpoint", "http://0.0.0.0:1234/v1", 'str')
        self.api_type = self.get_config_value("Slop_Command", "api_type", "chat", 'str')
        self.model = self.get_config_value("Slop_Command", "model", "gpt-4", 'str')
        self.api_key = self.get_config_value("Slop_Command", "api_key", "NA", 'str')
        self.max_tokens = self.get_config_value("Slop_Command", "max_tokens", 100, 'int')
        self.compact_threshold_tokens = self.get_config_value("Slop_Command", "compact_threshold_tokens", 1024, 'int')
        self._supports_max_completion_tokens = True
        self._response_ids: Dict[str, str] = {}
        self._compacted_contexts: Dict[str, str] = {}
        self._active_conversations: Set[str] = set()
        self._lock = None
        self._queue = asyncio.Queue()
        self._worker_task = None

    def _ensure_lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()

    def _ensure_worker_started(self):
        self._ensure_lock()
        if self._worker_task is None or self._worker_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._process_queue())
            except RuntimeError:
                pass

    def _get_conversation_key(self, message: MeshMessage) -> str:
        if message.is_dm:
            return f"dm:{message.sender_id}"
        return f"channel:{message.channel}"

    def matches_keyword(self, message: MeshMessage) -> bool:
        if self._check_mentions_ok(message.content):
            content = self._strip_mentions(message.content).strip()
            content_lower = content.lower()
            if content_lower.startswith("slop"):
                if len(content_lower) == 4 or content_lower[4] == ' ':
                    return True
        return False

    def matches_custom_syntax(self, message: MeshMessage) -> bool:
        if not self.enabled:
            return False
        if super().matches_custom_syntax(message):
            return True

        if self._is_bot_mentioned(message.content):
            return True

        content = message.content.strip()
        if not content:
            return False
        if message.is_dm:
            return True
        if self.matches_keyword(message):
            return True
        return False

    async def execute(self, message: MeshMessage) -> bool:
        self.logger.info("SlopCommand.execute called")
        self._ensure_worker_started()
        
        # Note: can_execute check is done by CommandManager before calling execute()
        # Cooldown is already recorded by CommandManager, so we don't call record_execution here

        content = message.content.strip()
        is_dm = message.is_dm
        conversation_key = self._get_conversation_key(message)
        reply_prefix = f"@[{self.bot.config.get('Bot', 'bot_name', fallback='Bot')}]"
        is_reply = content.startswith(reply_prefix)
        question = content

        if is_reply:
            question = question[len(reply_prefix):].strip()

        starts_with_slop = question.lower().startswith("slop") and (
            len(question.lower()) == 4 or question.lower()[4] == ' '
        )

        if starts_with_slop:
            if conversation_key in self._active_conversations:
                self._active_conversations.discard(conversation_key)
            if conversation_key in self._response_ids:
                del self._response_ids[conversation_key]
            if conversation_key in self._compacted_contexts:
                del self._compacted_contexts[conversation_key]
            question = question[4:].strip()

        if not question:
            if is_dm or is_reply:
                await self.send_response(message, "You need to ask a question!", skip_user_rate_limit=True)
                return True
            await self.send_response(message, "You need to ask a question! Usage: slop <new-question> or reply to continue the thread.", skip_user_rate_limit=True)
            return True

        self._ensure_lock()
        if self._lock.locked():
            self.logger.info("SlopCommand.execute: lock is locked, queueing and notifying user")
            await self.send_response(message, f"I'm busy with another request. You are number {self._queue.qsize() + 1} in line. Please wait.", skip_user_rate_limit=True)
            self.logger.info("SlopCommand.execute: finished notifying user")
        else:
            self.logger.info("SlopCommand.execute: lock is not locked, sending working message")
            await self.send_response(message, "💭...", skip_user_rate_limit=True)
            self.logger.info("SlopCommand.execute: finished sending working message")

        await self._queue.put((message, question, conversation_key, is_dm))
        self.logger.info(f"SlopCommand.execute: message put on queue. qsize is now {self._queue.qsize()}")
        
        return True

    def can_execute(self, message: MeshMessage) -> bool:
        self.logger.info(f"SlopCommand.can_execute: channel_allowed={self.is_channel_allowed(message)}, requires_dm={self.requires_dm}, is_dm={message.is_dm}, cooldown={self.cooldown_seconds}")
        result = super().can_execute(message)
        self.logger.info(f"SlopCommand.can_execute: super returned {result}, self.enabled={self.enabled}")
        if not result:
            return False
        return self.enabled

    async def _process_queue(self):
        self.logger.info("SlopCommand._process_queue worker started")
        while True:
            try:
                self.logger.info(f"SlopCommand._process_queue: waiting for message on queue. qsize is {self._queue.qsize()}")
                message, question, conversation_key, is_dm = await self._queue.get()
                self.logger.info("SlopCommand._process_queue: got message from queue")

                async with self._lock:
                    self.logger.info("SlopCommand._process_queue: lock acquired")
                    # Execution already recorded in execute() method to prevent cooldown issues
                    # self.record_execution(message.sender_id) 

                    if not is_dm:
                        self._active_conversations.add(conversation_key)

                    previous_response_id = self._response_ids.get(conversation_key)
                    self.logger.info(f"Conversation key: {conversation_key}, previous_response_id: {previous_response_id}")

                    try:
                        client = openai.AsyncOpenAI(
                            base_url=self.endpoint,
                            api_key=self.api_key,
                            timeout=45.0,
                        )

                        compacted_context = self._compacted_contexts.pop(conversation_key, None)
                        if compacted_context:
                            question = f"New question: {question}\n\nContext: {compacted_context}\n\n"
                            previous_response_id = None  # Clear previous response ID when using compacted context
                            self.logger.info(f"Added compacted context to question")

                        self.logger.info(f"Asking question: {question} | API Type: {self.api_type} | Model: {self.model}")

                        if self.api_type == "responses":
                            if not hasattr(client, 'responses'):
                                self.logger.error(f"Responses API not available. OpenAI version: {openai.__version__ if hasattr(openai, '__version__') else 'unknown'}")
                                await self.send_response(message, "Responses API not available. Update openai library.", skip_user_rate_limit=True)
                                continue
                            try:
                                extra_kwargs = {}
                                if self._supports_max_completion_tokens:
                                    extra_kwargs["max_completion_tokens"] = self.max_tokens
                                response = await client.responses.create(
                                    model=self.model,
                                    instructions=self.instructions,
                                    input=[{"type": "message", "role": "user", "content": f"[{message.sender_id}]: {question}", "name": message.sender_id}],
                                    **({"previous_response_id": previous_response_id} if previous_response_id else {}),
                                    **extra_kwargs,
                                )
                            except TypeError as e:
                                if "max_completion_tokens" in str(e):
                                    self.logger.warning(f"LM Studio doesn't support max_completion_tokens, disabling")
                                    self._supports_max_completion_tokens = False

                                    response = await client.responses.create(
                                        model=self.model,
                                        instructions=self.instructions,
                                        input=[{"type": "message", "role": "user", "content": f"[{message.sender_id}]: {question}", "name": message.sender_id}],
                                        **({"previous_response_id": previous_response_id} if previous_response_id else {}),
                                    )
                                else:
                                    raise

                            if hasattr(response, 'id') and response.id:
                                self.logger.info(f"Received response_id: {response.id}")
                                self._response_ids[conversation_key] = response.id
                            else:
                                self.logger.warning(f"No id in response. Response: {response}")

                            response_text = response.output[0].content[0].text
                            total_tokens = hasattr(response, 'usage') and getattr(response.usage, 'total_tokens', 0) or 0
                            self.logger.info(f"Received response: {response_text} | Total tokens: {total_tokens}")
                        else:
                            chat_completion = await client.chat.completions.create(
                                messages=[
                                    {
                                        "role": "user",
                                        "content": question,
                                    }
                                ],
                                model=self.model,
                                max_completion_tokens=self.max_tokens,
                            )
                            response_text = chat_completion.choices[0].message.content
                            total_tokens = 0

                        parsed = self._parse_response(response_text)
                        response_to_send = parsed["message"]

                        if parsed.get("more", {}).get("topics"):
                            topics = parsed["topics"]
                            response_to_send += f" | {', '.join(topics[:3])}"
                        
                        max_len = self.get_max_message_length(message)
                        if len(response_to_send) > max_len:
                            response_to_send = response_to_send[:max_len].rsplit(' ', 1)[0] + "..."

                        await self.send_response(message, response_to_send, skip_user_rate_limit=True)

                        if self.api_type == "responses" and total_tokens >= self.compact_threshold_tokens and self.compact_threshold_tokens > 0:
                            self.logger.info(f"Token count {total_tokens} exceeds threshold {self.compact_threshold_tokens}, compacting conversation")
                            compacted_text = await self._compact_conversation(client, previous_response_id)
                            if compacted_text:
                                self._compacted_contexts[conversation_key] = compacted_text
                                self.logger.info(f"Stored compacted context for conversation {conversation_key}")

                    except Exception as e:
                        self.logger.error(f"Error in slop command: {e}", exc_info=True)
                        await self.send_response(message, "Sorry, I couldn't get a response from the AI.", skip_user_rate_limit=True)
                    finally:
                        self.logger.info("SlopCommand._process_queue: lock released")
                        self._queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in _process_queue worker: {e}", exc_info=True)


    def _parse_response(self, response_text: str) -> dict[str, Any]:
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "message" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        
        suffixes = ['"}', '"]', '}', ']', '"', '"]}', ']}', '}"]', '}"}', '"]}']
        for suffix in suffixes:
            try:
                data = json.loads(response_text + suffix)
                if isinstance(data, dict) and "message" in data:
                    return {"message": data["message"] + "...", "truncated": True}
            except (json.JSONDecodeError, TypeError):
                pass
        
        return {"message": response_text}

    async def _compact_conversation(self, client, previous_response_id: Optional[str]) -> Optional[str]:
        try:
            self.logger.info("Compacting conversation...")

            if previous_response_id:
                extra_kwargs = {}
                if self._supports_max_completion_tokens:
                    extra_kwargs["max_completion_tokens"] = self.max_tokens
                response = await client.responses.create(
                    model=self.model,
                    instructions="Compact the conversation into a format you will be able to understand later on. Focus on keeping important details and context that would help you answer future questions related to this conversation. Be concise but retain key information and style.",
                    input=[
                        {"type": "message", "role": "user", "content": f"Compact the conversation"}
                    ],
                    previous_response_id=previous_response_id,
                    **extra_kwargs,
                )
                compacted_text = self._parse_response(response.output[0].content[0].text).get("message", "")
            else:
                compacted_text = f""
            return compacted_text
        except Exception as e:
            self.logger.error(f"Error compacting conversation: {e}", exc_info=True)
            return None

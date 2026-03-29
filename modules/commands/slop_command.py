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

    def __init__(self, bot):
        super().__init__(bot)
        self.enabled = self.get_config_value("Slop_Command", "enabled", True, 'bool')
        self.endpoint = self.get_config_value("Slop_Command", "endpoint", "http://0.0.0.0:1234/v1", 'str')
        self.api_type = self.get_config_value("Slop_Command", "api_type", "chat", 'str')
        self.model = self.get_config_value("Slop_Command", "model", "gpt-4", 'str')
        self.api_key = self.get_config_value("Slop_Command", "api_key", "NA", 'str')
        self.max_tokens = self.get_config_value("Slop_Command", "max_tokens", 100, 'int')
        self._response_ids: Dict[str, str] = {}
        self._active_conversations: Set[str] = set()

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

        content = message.content.strip()
        if not content:
            return False
        if message.is_dm:
            if not self.matches_keyword(message):
                return True
        content_lower = content.lower()
        if not content_lower.startswith("slop"):
            return True
        return False

    async def execute(self, message: MeshMessage) -> bool:
        if not self.enabled:
            return False

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
            question = question[4:].strip()

        if not question:
            if is_dm or is_reply:
                await self.send_response(message, "You need to ask a question!")
                return True
            await self.send_response(message, "You need to ask a question! Usage: slop <new-question> or reply to continue the thread.")
            return True

        self.record_execution(message.sender_id)

        if not is_dm:
            self._active_conversations.add(conversation_key)

        previous_response_id = self._response_ids.get(conversation_key)
        self.logger.info(f"Conversation key: {conversation_key}, previous_response_id: {previous_response_id}")

        try:
            client = openai.AsyncOpenAI(
                base_url=self.endpoint,
                api_key=self.api_key,
            )

            self.logger.info(f"Asking question: {question} | API Type: {self.api_type} | Model: {self.model}")

            if self.api_type == "responses":
                if not hasattr(client, 'responses'):
                    self.logger.error(f"Responses API not available. OpenAI version: {openai.__version__ if hasattr(openai, '__version__') else 'unknown'}")
                    await self.send_response(message, "Responses API not available. Update openai library.")
                    return True
                try:
                    response = await client.responses.create(
                        model=self.model,
                        max_completion_tokens=self.max_tokens,
                        input=[{"type": "message", "role": "user", "content": f"[{message.sender_id}]: {question}", "name": message.sender_id}],
                        **({"previous_response_id": previous_response_id} if previous_response_id else {}),
                    )
                except TypeError as e:
                    if "max_completion_tokens" in str(e):
                        self.logger.warning(f"LM Studio doesn't support max_completion_tokens, retrying without it")

                        response = await client.responses.create(
                            model=self.model,
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
            
            parsed = self._parse_response(response_text)
            response_to_send = parsed["message"]
            
            if parsed.get("more", {}).get("topics"):
                topics = parsed["topics"]
                response_to_send += f" | {', '.join(topics[:3])}"
            
            max_len = self.get_max_message_length(message)
            if len(response_to_send) > max_len:
                response_to_send = response_to_send[:max_len].rsplit(' ', 1)[0] + "..."

            await self.send_response(message, response_to_send)

        except Exception as e:
            self.logger.error(f"Error in slop command: {e}")
            await self.send_response(message, "Sorry, I couldn't get a response from the AI.")
        
        return True

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

    def can_execute(self, message: MeshMessage) -> bool:
        if not super().can_execute(message):
            return False
        return self.enabled

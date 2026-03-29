#!/usr/bin/env python3
"""
Integration tests for slop command with LM Studio endpoint.
Requires LM Studio running at the configured endpoint.
"""

import pytest
import asyncio
import socket
from modules.commands.slop_command import SlopCommand
from modules.models import MeshMessage


def is_lm_studio_available(host="host.docker.internal", port=1234, timeout=2):
    """Check if LM Studio is available at the given host/port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.error, socket.timeout):
        return False


lm_studio_available = pytest.mark.skipif(
    not is_lm_studio_available(),
    reason="LM Studio not running at host.docker.internal:1234"
)


@pytest.mark.integration
@lm_studio_available
class TestSlopCommandIntegration:
    """Integration tests for slop command with real LM Studio endpoint."""

    @pytest.fixture
    def slop_command(self, mock_bot):
        """Create a slop command instance."""
        command = SlopCommand(mock_bot)
        command.endpoint = "http://host.docker.internal:1234/v1"
        command.api_type = "chat"
        command.model = "qwen/qwen3.5-9b"
        command.api_key = "NA"
        command.max_tokens = 50
        return command

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Chat API may not support this model in LM Studio - use responses API")
    async def test_chat_api_integration(self, slop_command):
        """Test slop command with chat completions API against LM Studio."""
        message = MeshMessage(
            content="slop What is 1+1?",
            sender_id="test_sender",
            is_dm=False
        )

        result = await slop_command.execute(message)
        
        assert result is True
        assert slop_command.bot.command_manager.send_response.called
        
        call_args = slop_command.bot.command_manager.send_response.call_args
        response_text = call_args[0][1]
        assert len(response_text) > 0, f"Empty response from chat API"
        assert "couldn't get a response" not in response_text.lower()

    @pytest.mark.asyncio
    async def test_responses_api_integration(self, mock_bot):
        """Test slop command with responses API against LM Studio."""
        command = SlopCommand(mock_bot)
        command.endpoint = "http://host.docker.internal:1234/v1"
        command.api_type = "responses"
        command.model = "qwen/qwen3.5-9b"
        command.api_key = "NA"
        command.max_tokens = 50
        
        message = MeshMessage(
            content="slop What is 2+2?",
            sender_id="test_sender",
            is_dm=False
        )

        result = await command.execute(message)
        
        assert result is True
        assert command.bot.command_manager.send_response.called
        
        call_args = command.bot.command_manager.send_response.call_args
        response_text = call_args[0][1]
        assert len(response_text) > 0

    @pytest.mark.asyncio
    async def test_json_response_parsing(self, mock_bot):
        """Test that JSON schema responses are parsed correctly."""
        command = SlopCommand(mock_bot)
        command.endpoint = "http://host.docker.internal:1234/v1"
        command.api_type = "responses"
        command.model = "qwen/qwen3.5-9b"
        command.api_key = "NA"
        command.max_tokens = 50
        
        message = MeshMessage(
            content="slop List one fruit",
            sender_id="test_sender",
            is_dm=False
        )

        await command.execute(message)
        
        assert command.bot.command_manager.send_response.called
        
        call_args = command.bot.command_manager.send_response.call_args
        response_text = call_args[0][1]
        assert len(response_text) > 0

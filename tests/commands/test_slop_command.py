#!/usr/bin/env python3
"""
Tests for the SlopCommand class.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from modules.commands.slop_command import SlopCommand
from modules.models import MeshMessage

@pytest.fixture
def bot():
    """Fixture for a mock bot."""
    bot = MagicMock()
    bot.config.has_section.return_value = True
    bot.config.has_option.return_value = True
    bot.config.get.side_effect = lambda section, key, fallback=None: {
        ("Slop_Command", "enabled"): True,
        ("Slop_Command", "endpoint"): "http://0.0.0.0:1234/v1",
        ("Slop_Command", "model"): "gpt-4",
        ("Slop_Command", "api_key"): "NA",
        ("Slop_Command", "max_tokens"): 100,
    }.get((section, key), fallback)
    bot.config.getboolean.side_effect = lambda section, key, fallback=None: {
        ("Slop_Command", "enabled"): True,
    }.get((section, key), fallback)
    bot.config.getint.side_effect = lambda section, key, fallback=None: {
        ("Slop_Command", "max_tokens"): 100,
    }.get((section, key), fallback)
    bot.logger = MagicMock()
    bot.command_manager.send_response = AsyncMock(return_value=True)
    return bot

@pytest.mark.asyncio
async def test_slop_command_execute_success(bot):
    """Test the successful execution of the slop command."""
    # Arrange
    slop_command = SlopCommand(bot)
    message = MeshMessage(
        content="slop What is the meaning of life?",
        sender_id="test_sender",
        is_dm=False
    )

    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [MagicMock()]
    mock_chat_completion.choices[0].message.content = "42"

    with patch('modules.commands.slop_command.openai.AsyncOpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)
        mock_openai_class.return_value = mock_client

        # Act
        await slop_command.execute(message)

        # Assert
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[{'role': 'user', 'content': 'What is the meaning of life?'}],
            model='gpt-4',
            max_completion_tokens=100
        )
        bot.command_manager.send_response.assert_called_once_with(message, "42", skip_user_rate_limit=False)

@pytest.mark.asyncio
async def test_slop_command_truncates_long_response(bot):
    """Test that the slop command truncates long responses."""
    # Arrange
    slop_command = SlopCommand(bot)
    slop_command.get_max_message_length = MagicMock(return_value=50)
    message = MeshMessage(
        content="slop Tell me a very long story",
        sender_id="test_sender",
        is_dm=False
    )

    long_response = "This is a very long story that will certainly exceed the maximum message length."
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [MagicMock()]
    mock_chat_completion.choices[0].message.content = long_response

    with patch('modules.commands.slop_command.openai.AsyncOpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)
        mock_openai_class.return_value = mock_client

        # Act
        await slop_command.execute(message)

        # Assert
        expected_response = "This is a very long story that will certainly..."
        bot.command_manager.send_response.assert_called_once_with(message, expected_response, skip_user_rate_limit=False)


@pytest.mark.asyncio
async def test_slop_command_no_question(bot):
    """Test the slop command with no question."""
    # Arrange
    slop_command = SlopCommand(bot)
    message = MeshMessage(content="slop", sender_id="test_sender", is_dm=False)

    # Act
    await slop_command.execute(message)

    # Assert
    bot.command_manager.send_response.assert_called_once_with(
        message, "You need to ask a question! Usage: slop <question>", skip_user_rate_limit=False
    )

@pytest.mark.asyncio
async def test_slop_command_disabled(bot):
    """Test that the slop command does not execute when disabled."""
    # Arrange
    bot.config.get.side_effect = lambda section, key, fallback=None: {
        ("Slop_Command", "enabled"): False,
    }.get((section, key), fallback)
    bot.config.getboolean.side_effect = lambda section, key, fallback=None: {
        ("Slop_Command", "enabled"): False,
    }.get((section, key), fallback)
    slop_command = SlopCommand(bot)
    message = MeshMessage(
        content="slop What is the meaning of life?",
        sender_id="test_sender",
        is_dm=False
    )

    # Act
    result = await slop_command.execute(message)

    # Assert
    assert result is False
    bot.command_manager.send_response.assert_not_called()

@pytest.mark.asyncio
async def test_slop_command_api_error(bot):
    """Test the slop command when the API call fails."""
    # Arrange
    slop_command = SlopCommand(bot)
    message = MeshMessage(
        content="slop What is the meaning of life?",
        sender_id="test_sender",
        is_dm=False
    )

    with patch('modules.commands.slop_command.openai.AsyncOpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        mock_openai_class.return_value = mock_client

        # Act
        await slop_command.execute(message)

        # Assert
        bot.command_manager.send_response.assert_called_once_with(
            message, "Sorry, I couldn't get a response from the AI.", skip_user_rate_limit=False
        )
        bot.logger.error.assert_called_once_with("Error in slop command: API Error")


@pytest.mark.asyncio
async def test_slop_command_truncated_json_recovers_message(bot):
    """Test that truncated JSON still recovers the message content."""
    slop_command = SlopCommand(bot)
    
    truncated_json = '{"message": "This is a long'
    result = slop_command._parse_response(truncated_json)
    assert result["message"] == "This is a long..."
    assert result.get("truncated") is True
    
    truncated_json_with_topics = '{"message": "Hello world", "topics": ["a", "b'
    result = slop_command._parse_response(truncated_json_with_topics)
    assert result["message"] == "Hello world..."
    assert result.get("truncated") is True


@pytest.mark.asyncio
async def test_slop_command_parse_response_with_escaped_quotes(bot):
    """Test that JSON with escaped quotes is parsed correctly."""
    slop_command = SlopCommand(bot)
    
    response_with_escaped_quotes = '{"message": "She said \\"hello\\" to me", "topics": ["greeting"]}'
    result = slop_command._parse_response(response_with_escaped_quotes)
    assert result["message"] == 'She said "hello" to me'
    
    truncated_with_escaped_quotes = '{"message": "Test \\"quote'
    result = slop_command._parse_response(truncated_with_escaped_quotes)
    assert result["message"] == 'Test "quote...'
    assert result.get("truncated") is True
    
    truncated_without_closing_brace = '{"message":"Once upon a time in Newcastle, Australia, there was a little kangaroo named Kevin'
    result = slop_command._parse_response(truncated_without_closing_brace)
    assert result["message"] == "Once upon a time in Newcastle, Australia, there was a little kangaroo named Kevin..."
    assert result.get("truncated") is True


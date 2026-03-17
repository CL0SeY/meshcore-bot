#!/usr/bin/env python3
"""
Local LLM Chatbot command for the MeshCore Bot
Allows users to chat with a local LLM using the picolm CLI
"""

import os
import asyncio
from typing import Any
from .base_command import BaseCommand
from ..models import MeshMessage

class ChatCommand(BaseCommand):
    """Handles chatbot interactions using the picolm CLI"""
    
    # Plugin metadata
    name = "chat"
    keywords = ['chat', 'ask', 'llm', 'picolm']
    description = "Chat with a local AI using picolm"
    category = "ai"
    
    # Documentation
    short_description = "Chat with a local AI"
    usage = "chat <your message>"
    examples = ["chat what is the capital of France?", "ask tell me a joke"]
    
    def __init__(self, bot: Any):
        """Initialize the chat command."""
        super().__init__(bot)
        
        # Load configuration
        self.enabled = self.get_config_value('Chat', 'enabled', fallback=False, value_type='bool')
        self.model_path = self.get_config_value('Chat', 'model_path', fallback="")
        self.max_tokens = self.get_config_value('Chat', 'max_tokens', fallback=150, value_type='int')
        
    def get_help_text(self) -> str:
        """Get help text for the command."""
        return "Chat with a local AI. Usage: chat <your message>"
    
    def can_execute(self, message: MeshMessage) -> bool:
        """Check if this command can be executed."""
        if not self.enabled:
            return False
        return super().can_execute(message)
    
    async def execute(self, message: MeshMessage) -> bool:
        """Execute the chatbot command."""
        # Extract the prompt
        parts = message.content.split(maxsplit=1)
        if len(parts) < 2:
            return await self.send_response(message, "Please provide a message. Example: chat hello")
            
        prompt = parts[1].strip()
        
        try:
            self.logger.info(f"Generating response via picolm for: {prompt}")
            
            # Build command
            cmd = ['picolm']
            if self.model_path and os.path.exists(self.model_path):
                cmd.extend(['-m', self.model_path])
            if self.max_tokens:
                cmd.extend(['-n', str(self.max_tokens)])
            cmd.append(prompt)
            
            # Execute picolm CLI asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                reply = stdout.decode().strip()
                if not reply:
                    reply = "I don't have a response for that."
                return await self.send_response(message, reply)
            else:
                err_msg = stderr.decode().strip()
                self.logger.error(f"picolm error: {err_msg}")
                return await self.send_response(message, "Sorry, I encountered an error while thinking.")
                
        except FileNotFoundError:
            self.logger.error("picolm CLI not found. Please install it and ensure it's in your PATH.")
            return await self.send_response(message, "Chatbot is not properly configured (picolm not found).")
        except Exception as e:
            self.logger.error(f"Error generating picolm response: {e}")
            return await self.send_response(message, "Sorry, I encountered an error while thinking.")


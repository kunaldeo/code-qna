"""Google Gemini AI client using the latest google-genai SDK."""

from google import genai
from google.genai import types
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-001", 
                 temperature: float = 0.3, max_output_tokens: int = 4000,
                 enable_thinking: bool = False, thinking_budget: Optional[int] = None,
                 include_thoughts: bool = False, vertexai: bool = False,
                 project: Optional[str] = None, location: Optional[str] = None,
                 http_options: Optional[types.HttpOptions] = None):
        load_dotenv()
        
        # Check if we should use Vertex AI from environment
        self.use_vertexai = vertexai or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
        
        if self.use_vertexai:
            # Vertex AI configuration
            self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            if not self.project:
                raise ValueError("Project ID not found. Set GOOGLE_CLOUD_PROJECT environment variable or pass project parameter")
            
            # Create client for Vertex AI
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                http_options=http_options
            )
        else:
            # Gemini Developer API configuration
            # Check for API key in this order: parameter, GEMINI_API_KEY, GOOGLE_API_KEY
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            
            # Warn if both are set
            if os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
                print("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GEMINI_API_KEY.")
            elif os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
                print("Using GOOGLE_API_KEY for Gemini client.")
            
            if not self.api_key:
                raise ValueError("API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
            
            # Create client for Developer API
            self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.include_thoughts = include_thoughts
        
        # System instruction for code Q&A
        self.system_instruction = """You are a code analysis assistant specialized in understanding and explaining source code.
When answering questions about code:
1. Be precise and reference specific files, functions, and line numbers when possible
2. Explain the code's purpose, how it works, and its relationships to other parts
3. If asked about functionality, trace through the execution flow
4. Highlight important patterns, dependencies, and potential issues
5. Keep answers focused and relevant to the question
"""

    def generate_answer(self, question: str, context: Dict[str, Any]) -> str:
        """Generate an answer based on the question and extracted context."""
        # Build the prompt with context
        prompt = self._build_prompt(question, context)
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            
            # Add thinking configuration if enabled
            if self.enable_thinking:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=self.include_thoughts
                )
                if self.thinking_budget is not None:
                    thinking_config.thinking_budget = self.thinking_budget
                config.thinking_config = thinking_config
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            
            # If include_thoughts is enabled, format the response with thoughts
            if self.include_thoughts and hasattr(response, 'candidates') and response.candidates:
                result_parts = []
                for part in response.candidates[0].content.parts:
                    if not part.text:
                        continue
                    if hasattr(part, 'thought') and part.thought:
                        result_parts.append(f"\n*💭 {part.text}*\n")
                    else:
                        result_parts.append(part.text)
                return "\n".join(result_parts)
            else:
                # Return just the answer without thoughts
                result_text = ""
                if hasattr(response, 'candidates') and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.text and not (hasattr(part, 'thought') and part.thought):
                            result_text += part.text
                return result_text if result_text else response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_answer_stream(self, question: str, context: Dict[str, Any]):
        """Generate an answer with streaming for real-time output."""
        prompt = self._build_prompt(question, context)
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            
            # Add thinking configuration if enabled
            if self.enable_thinking:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=self.include_thoughts
                )
                if self.thinking_budget is not None:
                    thinking_config.thinking_budget = self.thinking_budget
                config.thinking_config = thinking_config
            
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=config,
            ):
                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if not part.text:
                                continue
                            elif hasattr(part, 'thought') and part.thought:
                                if self.include_thoughts:
                                    yield f"\n*💭 {part.text}*\n"
                            else:
                                yield part.text
        except Exception as e:
            yield f"Error generating response: {str(e)}"

    def _build_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """Build the prompt with question and context."""
        prompt_parts = [
            f"Question: {question}\n\n",
            "Here is the relevant code context to answer the question:\n\n"
        ]
        
        # Add file contexts
        if "context_files" in context and "files" in context["context_files"]:
            for file_info in context["context_files"]["files"]:
                prompt_parts.append(f"\n{'='*80}\n")
                prompt_parts.append(f"File: {file_info['path']}\n")
                prompt_parts.append(f"Language: {file_info.get('language', 'unknown')}\n")
                
                if file_info.get('matches'):
                    prompt_parts.append(f"Matches found: {file_info['matches']}\n")
                
                if file_info.get('is_partial'):
                    prompt_parts.append("(Showing partial content with context around matches)\n")
                
                prompt_parts.append(f"\n{file_info['content']}\n")
        
        # Add search keywords used
        if "keywords" in context:
            prompt_parts.append(f"\n\nSearch keywords used: {', '.join(context['keywords'])}\n")
        
        prompt_parts.append("\n\nBased on the code context above, please provide a comprehensive answer to the question.")
        
        return "".join(prompt_parts)

    def generate_answer_with_thoughts(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an answer with thinking details including thought summaries and token counts."""
        prompt = self._build_prompt(question, context)
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            
            # Force thinking configuration for this method
            thinking_config = types.ThinkingConfig(
                include_thoughts=True
            )
            if self.thinking_budget is not None:
                thinking_config.thinking_budget = self.thinking_budget
            config.thinking_config = thinking_config
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            
            result = {
                "answer": "",
                "thoughts": "",
                "thoughts_token_count": 0,
                "output_token_count": 0,
                "total_token_count": 0
            }
            
            # Extract thoughts and answer from response parts
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        if hasattr(part, 'thought') and part.thought:
                            result["thoughts"] += f"*💭 {part.text}*\n"
                        else:
                            result["answer"] += part.text
            else:
                result["answer"] = response.text
            
            # Extract token counts if available
            if hasattr(response, 'usage_metadata'):
                if hasattr(response.usage_metadata, 'thoughts_token_count'):
                    result["thoughts_token_count"] = response.usage_metadata.thoughts_token_count
                if hasattr(response.usage_metadata, 'candidates_token_count'):
                    result["output_token_count"] = response.usage_metadata.candidates_token_count
                if hasattr(response.usage_metadata, 'total_token_count'):
                    result["total_token_count"] = response.usage_metadata.total_token_count
            
            return result
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "thoughts": "",
                "thoughts_token_count": 0,
                "output_token_count": 0,
                "total_token_count": 0
            }

    def create_chat_session(self) -> 'ChatSession':
        """Create a chat session for multi-turn conversations."""
        return ChatSession(self)


class ChatSession:
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        # Create chat configuration with thinking support
        config = types.GenerateContentConfig(
            system_instruction=self.client.system_instruction,
            temperature=self.client.temperature,
            max_output_tokens=self.client.max_output_tokens,
        )
        
        # Add thinking configuration if enabled
        if self.client.enable_thinking:
            thinking_config = types.ThinkingConfig(
                include_thoughts=self.client.include_thoughts
            )
            if self.client.thinking_budget is not None:
                thinking_config.thinking_budget = self.client.thinking_budget
            config.thinking_config = thinking_config
        
        # Create chat using the correct API pattern
        self.chat = self.client.client.chats.create(
            model=self.client.model,
            config=config
        )
        self.history = []

    def ask(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Ask a question in the chat session."""
        if context:
            # Include context in the question
            message = self.client._build_prompt(question, context)
        else:
            message = question
        
        response = self.chat.send_message(message)
        
        # Process response to handle thoughts
        response_text = ""
        if self.client.include_thoughts and hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if hasattr(part, 'thought') and part.thought:
                    response_text += f"\n*💭 {part.text}*\n"
                else:
                    response_text += part.text
        else:
            # Return just the answer without thoughts
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.text and not (hasattr(part, 'thought') and part.thought):
                        response_text += part.text
            else:
                response_text = response.text
        
        self.history.append({
            "role": "user",
            "content": question,  # Store original question
        })
        self.history.append({
            "role": "assistant",
            "content": response_text
        })
        
        return response_text

    def ask_stream(self, question: str, context: Optional[Dict[str, Any]] = None):
        """Ask a question with streaming response."""
        if context:
            message = self.client._build_prompt(question, context)
        else:
            message = question
        
        full_response = []
        for chunk in self.chat.send_message_stream(message):
            if chunk.candidates and len(chunk.candidates) > 0:
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if not part.text:
                            continue
                        elif hasattr(part, 'thought') and part.thought:
                            if self.client.include_thoughts:
                                text = f"\n*💭 {part.text}*\n"
                                full_response.append(text)
                                yield text
                        else:
                            full_response.append(part.text)
                            yield part.text
        
        # Update history after streaming completes
        self.history.append({
            "role": "user",
            "content": question,
        })
        self.history.append({
            "role": "assistant",
            "content": "".join(full_response)
        })

    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.history
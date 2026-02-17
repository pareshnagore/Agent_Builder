# import ollama
# from google import genai
# from core.config import Config


# class LLMClient:

#     def __init__(self):
#         self.gemini_client = None

#         if Config.GEMINI_API_KEY:
#             self.gemini_client = genai.Client(
#                 api_key=Config.GEMINI_API_KEY
#             )

#     # -------- OLLAMA --------
#     def chat_ollama(self, model, messages):
#         response = ollama.chat(
#             model=model,
#             messages=messages
#         )
#         return response["message"]["content"]

#     # -------- GEMINI --------
#     def list_gemini_models(self):
#         try:
#             models = self.gemini_client.models.list()
#             return [m.name for m in models]
#         except:
#             return ["gemini-1.5-flash"]

#     def chat_gemini(self, model, messages):

#         conversation = ""
#         for m in messages:
#             conversation += f"{m['role']}: {m['content']}\n"

#         response = self.gemini_client.models.generate_content(
#             model=model,
#             contents=conversation
#         )

#         return response.text


"""
LLM Adapter for Agent_ng.
Supports both Ollama and Google Gemini with a unified interface.
"""

import os
from typing import Optional, Literal
from abc import ABC, abstractmethod
import requests
import json

from core.config import Config


class LLMException(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available models for this provider."""
        pass

    @abstractmethod
    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        """Send a chat request and return the assistant's response."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM Provider."""

    def __init__(self, host: str = None, api_key: str = None, use_cloud: bool = False):
        # Determine if using cloud or local
        if use_cloud and Config.OLLAMA_CLOUD_ENABLED:
            self.host = Config.OLLAMA_CLOUD_HOST
            self.api_key = Config.OLLAMA_API_KEY
            self.is_cloud = True
        else:
            self.host = host or Config.OLLAMA_HOST
            self.api_key = None
            self.is_cloud = False
        
        if not self.host.endswith("/"):
            self.host += "/"

    def list_models(self) -> list[str]:
        """Fetch available models from Ollama."""
        try:
            headers = {}
            if self.is_cloud and self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # print(f"Fetching models from Ollama at {self.host}api/tags")
            response = requests.get(f"{self.host}api/tags", headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except requests.RequestException as e:
            raise LLMException(f"Failed to fetch Ollama models: {str(e)}")
        except Exception as e:
            raise LLMException(f"Unexpected error fetching Ollama models: {str(e)}")

    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        # Add Authorization header if cloud
        headers = {}
        if self.is_cloud and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        """Send a chat request to Ollama."""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            # Merge any additional options
            payload.update(kwargs)

            url = f"{self.host}api/chat"
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Better error handling
            if response.status_code != 200:
                error_text = response.text
                try:
                    error_json = response.json()
                    error_text = error_json.get("error", error_text)
                except:
                    pass
                raise LLMException(f"Ollama API error ({response.status_code}): {error_text}")
            
            data = response.json()
            return data.get("message", {}).get("content", "No response")
        except requests.RequestException as e:
            raise LLMException(f"Ollama chat request failed: {str(e)}")
        except Exception as e:
            raise LLMException(f"Unexpected error in Ollama chat: {str(e)}")

    def chat_stream(self, model: str, messages: list[dict], **kwargs):
        """Send a chat request to Ollama and stream the response."""
        try:
            headers = {}
            if self.is_cloud and self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,  # Enable streaming
            }
            payload.update(kwargs)
    
            url = f"{self.host}api/chat"
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30,
                stream=True  # Important for streaming
            )
            
            if response.status_code != 200:
                raise LLMException(f"Ollama API error ({response.status_code}): {response.text}")
            
            # Stream response line by line
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                        
        except requests.RequestException as e:
            raise LLMException(f"Ollama chat stream failed: {str(e)}")
        except Exception as e:
            raise LLMException(f"Unexpected error in Ollama chat stream: {str(e)}")


class GeminiProvider(LLMProvider):
    """Google Gemini LLM Provider."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        if not self.api_key:
            raise LLMException("GEMINI_API_KEY not set in environment or config")
        
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise LLMException("google-genai not installed. Install with: pip install google-genai")
        except Exception as e:
            raise LLMException(f"Failed to initialize Gemini client: {str(e)}")

    def list_models(self) -> list[str]:
        """Fetch available Gemini models dynamically from the API."""
        try:
            models = self.client.models.list()
            model_names = []
            for model in models:
                # Extract model name from the full identifier
                name = model.name.split("/")[-1] if hasattr(model, "name") else str(model)
                # Filter to include only generative models
                if hasattr(model, "supported_generation_methods"):
                    if "generateContent" in model.supported_generation_methods:
                        model_names.append(name)
                else:
                    # Fallback: include if it looks like a valid model name
                    if name and not name.startswith("_"):
                        model_names.append(name)
            
            # If no models found from API, return fallback list
            if not model_names:
                model_names = [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                ]
            
            return model_names
        except Exception as e:
            # Fallback to known models if API fails
            return [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]

    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        """Send a chat request to Gemini."""
        try:
            from google.genai import types
            
            # Extract system prompt if present
            system_prompt = None
            content_messages = []
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                
                if not role or not content:
                    continue
                
                if role == "system":
                    system_prompt = content
                else:
                    # Convert "assistant" to "model" for Gemini
                    gemini_role = "model" if role == "assistant" else role
                    content_messages.append({
                        "role": gemini_role,
                        "parts": [{"text": content}]
                    })
            
            if not content_messages:
                raise LLMException("No valid messages to send to Gemini")
            
            # Create config with system instruction if present
            config_dict = {}
            if system_prompt:
                config_dict["system_instruction"] = system_prompt
            
            # Add any additional kwargs
            config_dict.update(kwargs)
            
            config = types.GenerateContentConfig(**config_dict) if config_dict else None
            
            # Send request directly using generate_content
            response = self.client.models.generate_content(
                model=model,
                contents=content_messages,
                config=config
            )
            
            return response.text
            
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"Gemini chat request failed: {str(e)}")

    def chat_stream(self, model: str, messages: list[dict], **kwargs):
        """Send a chat request to Gemini and stream the response."""
        try:
            from google.genai import types
            
            system_prompt = None
            content_messages = []
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                
                if not role or not content:
                    continue
                
                if role == "system":
                    system_prompt = content
                else:
                    gemini_role = "model" if role == "assistant" else role
                    content_messages.append({
                        "role": gemini_role,
                        "parts": [{"text": content}]
                    })
            
            if not content_messages:
                raise LLMException("No valid messages to send to Gemini")
            
            config_dict = {}
            if system_prompt:
                config_dict["system_instruction"] = system_prompt
            config_dict.update(kwargs)
            
            config = types.GenerateContentConfig(**config_dict) if config_dict else None
            
            # Stream response
            response = self.client.models.generate_content(
                model=model,
                contents=content_messages,
                config=config,
                stream=True  # Enable streaming
            )
            
            # Yield text chunks as they arrive
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"Gemini chat stream failed: {str(e)}")

def chat_stream(
    self,
    provider: Literal["ollama", "ollama-cloud", "gemini"],
    model: str,
    messages: list[dict],
    **kwargs
):
    """
    Send a chat request and stream the response.
    
    Yields response text chunks as they arrive.
    """
    if provider not in self.providers or self.providers[provider] is None:
        raise LLMException(f"Provider '{provider}' not available or not configured")
    
    return self.providers[provider].chat_stream(model, messages, **kwargs)

def chat_ollama_stream(self, model: str, messages: list[dict], **kwargs):
    """Convenience method for streaming Ollama chat."""
    return self.chat_stream("ollama", model, messages, **kwargs)

def chat_ollama_cloud_stream(self, model: str, messages: list[dict], **kwargs):
    """Convenience method for streaming Ollama Cloud chat."""
    return self.chat_stream("ollama-cloud", model, messages, **kwargs)

def chat_gemini_stream(self, model: str, messages: list[dict], **kwargs):
    """Convenience method for streaming Gemini chat."""
    return self.chat_stream("gemini", model, messages, **kwargs)
class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    Usage:
        llm = LLMClient()
        models = llm.list_models("ollama")
        reply = llm.chat("ollama", "gemma2:2b", messages)
    """

    def __init__(self):
        self.providers = {
            "ollama": OllamaProvider(use_cloud=False),
            "ollama-cloud": OllamaProvider(use_cloud=True) if Config.OLLAMA_CLOUD_ENABLED else None,
            "gemini": GeminiProvider() if Config.GEMINI_API_KEY else None,
        }

    def list_models(self, provider: Literal["ollama", "ollama-cloud", "gemini"]) -> list[str]:
        """Get available models for a provider."""
        if provider not in self.providers or self.providers[provider] is None:
            raise LLMException(f"Provider '{provider}' not available or not configured")
        return self.providers[provider].list_models()

    def chat(
        self,
        provider: Literal["ollama", "ollama-cloud", "gemini"],
        model: str,
        messages: list[dict],
        **kwargs
    ) -> str:
        """
        Send a chat request to the specified provider.
        
        Args:
            provider: "ollama", "ollama-cloud" or "gemini"
            model: Model name (e.g., "gemma2:2b" for Ollama, "gemini-2.0-flash" for Gemini)
            messages: List of message dicts with "role" and "content" keys
            **kwargs: Additional options (e.g., temperature, top_p)
        
        Returns:
            Assistant's response text
        
        Raises:
            LLMException: If provider not available or request fails
        """
        if provider not in self.providers or self.providers[provider] is None:
            raise LLMException(f"Provider '{provider}' not available or not configured")
        
        return self.providers[provider].chat(model, messages, **kwargs)

    def list_ollama_models(self) -> list[str]:
        """Convenience method to list Ollama models."""
        return self.list_models("ollama")

    def list_ollama_cloud_models(self) -> list[str]:
        """Convenience method to list Ollama Cloud models."""
        return self.list_models("ollama-cloud")

    def list_gemini_models(self) -> list[str]:
        """Convenience method to list Gemini models."""
        return self.list_models("gemini")

    def chat_ollama(self, model: str, messages: list[dict], **kwargs) -> str:
        """Convenience method for Ollama chat."""
        return self.chat("ollama", model, messages, **kwargs)
    
    def chat_ollama_cloud(self, model: str, messages: list[dict], **kwargs) -> str:
        """Convenience method for Ollama Cloud chat."""
        return self.chat("ollama-cloud", model, messages, **kwargs)

    def chat_gemini(self, model: str, messages: list[dict], **kwargs) -> str:
        """Convenience method for Gemini chat."""
        return self.chat("gemini", model, messages, **kwargs)
    
    def chat_stream(
        self,
        provider: Literal["ollama", "ollama-cloud", "gemini"],
        model: str,
        messages: list[dict],
        **kwargs
    ):
        """
        Send a chat request and stream the response.
        
        Yields response text chunks as they arrive.
        """
        if provider not in self.providers or self.providers[provider] is None:
            raise LLMException(f"Provider '{provider}' not available or not configured")
        
        return self.providers[provider].chat_stream(model, messages, **kwargs)
    
    def chat_ollama_stream(self, model: str, messages: list[dict], **kwargs):
        """Convenience method for streaming Ollama chat."""
        return self.chat_stream("ollama", model, messages, **kwargs)
    
    def chat_ollama_cloud_stream(self, model: str, messages: list[dict], **kwargs):
        """Convenience method for streaming Ollama Cloud chat."""
        return self.chat_stream("ollama-cloud", model, messages, **kwargs)
    
    def chat_gemini_stream(self, model: str, messages: list[dict], **kwargs):
        """Convenience method for streaming Gemini chat."""
        return self.chat_stream("gemini", model, messages, **kwargs)
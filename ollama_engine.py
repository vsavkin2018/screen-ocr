import httpx
import io
from typing import List, AsyncGenerator, Tuple, Optional, Dict
import base64
import asyncio
import json

from ocr_base import *

class OllamaEngine(BaseEngine):
    def __init__(self, config: dict):
        self.config = config["ollama"]
        self._active_request: Optional[httpx.Response] = None
        self._cancelled = False

    def get_max_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        return self.config.get("max_width", 1200), self.config.get("max_height", 1200)

    async def cancel(self):
        """Cancel current OCR operation"""
        self._cancelled = True
        if self._active_request:
            # Close the HTTP connection immediately
            await self._active_request.aclose()
            self._active_request = None

    async def stream_ocr(self, image: Image.Image) -> AsyncGenerator[str, None]:
        """Ollama-specific OCR implementation with error handling"""
        self._cancelled = False
        try:
            async with httpx.AsyncClient() as client:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                b64_image = base64.b64encode(buffer.getvalue()).decode()

                if not self.config["current_prompt"]:
                    self.config["current_prompt"] = self.config["prompt_list"][0][1]

                data = {
                    "model": self.config["model"],
                    "prompt": self.config["current_prompt"],
                    "images": [b64_image],
                    "stream": True
                }

                try:
                    self._active_request = client.stream(
                        "POST",
                        self.config["base_url"],
                        json=data,
                        timeout=self.config.get("timeout", 180)
                    )

                    async with self._active_request as response:
                        if response.status_code != 200:
                            yield f"\n OCR Error ({response.status_code}): {await response.atext()}"
                            return

                        async for chunk in response.aiter_lines():
                            if self._cancelled:
                                yield "\n OCR aborted"
                                return
                            try:
                                yield json.loads(chunk).get("response", "")
                            except json.JSONDecodeError:
                                continue
                except httpx.ReadTimeout:
                    yield "\n OCR Error: Request timed out. Check connection or increase timeout."
                except httpx.ConnectError:
                    yield "\n OCR Error: Cannot connect to Ollama. Is it running?"
                except httpx.HTTPStatusError as e:
                    yield f"\n OCR Error: {e.response.status_code} {e.response.reason_phrase}"
                except Exception as e:
                    yield f"\n Unexpected OCR Error: {str(e)}"
        finally:
            self._active_request = None
            self._cancelled = False

    async def stream_chat(self, message: str, context: dict) -> AsyncGenerator[str, None]:
        """Multimodal chat with image context"""
        self._cancelled = False
        try:
            async with httpx.AsyncClient() as client:
                # Reuse prepared image from OCR context
                img = context['prepared_image']
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                b64_image = base64.b64encode(buffer.getvalue()).decode()

                # Build multimodal prompt
                prompt = self.config["chat_prompt"].format(
                        ocr_text=context['ocr_text'],
                        question=message,
                )

                data = {
                    "model": self.config["model"],
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": True
                }

                try:
                    self._active_request = client.stream(
                        "POST",
                        self.config["base_url"],
                        json=data,
                        timeout=self.config.get("timeout", 180)
                    )

                    async with self._active_request as response:
                        if response.status_code != 200:
                            yield f"\nChat Error ({response.status_code}): {await response.atext()}"
                            return

                        async for chunk in response.aiter_lines():
                            if self._cancelled:
                                yield "\n Chat aborted"
                                return
                            try:
                                yield json.loads(chunk).get("response", "")
                            except json.JSONDecodeError:
                                continue
                except httpx.RequestError as e:
                    print(f"DEBUG: Caught httpx.RequestError in stream_chat: {type(e)}")
                    if self._cancelled:
                        yield "\n Chat aborted"
                    else:
                        yield "\n Chat connection error"
                    return
                except asyncio.CancelledError as e:
                    print(f"DEBUG: Caught asyncio.CancelledError in stream_chat: {type(e)}")
                    yield "\n Chat aborted"
                    return



        except Exception as e:
            yield f"\nChat Error: {str(e)}"
        finally:
            self._active_request = None
            self._cancelled = False

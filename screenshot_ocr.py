#!/usr/bin/env python3
import os
import sys
import base64
import asyncio
import json
from pathlib import Path
from typing import List, AsyncGenerator, Tuple

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.enums import EditingMode

from PIL import Image
import io
import subprocess
import signal

# ======================
# CONFIGURATION CONSTANTS
# ======================
IMAGE_SETTINGS = {
    "default_directory": Path.home() / "Pictures" / "Screenshots",
    "filename_pattern": "Screenshot*.png"
}

OLLAMA_SETTINGS = {
    "base_url": "http://localhost:11434/api/generate",
    "vision_model": "llama3.2-vision"
}

PREVIEW_SETTINGS = {
    "max_width": 800,
    "max_height": 600
}

OCR_SETTINGS = {
    "max_width": 1200,
    "max_height": 1200
}

# ==================
# APPLICATION SETUP
# ==================

class ImageExplorer:
    def __init__(self, image_dir: Path = None, reversed: bool = False):
        default_dir = IMAGE_SETTINGS["default_directory"]
        if not default_dir.is_dir():
            print(f"Creating default directory: {default_dir}")
            default_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = image_dir or default_dir
        image_paths = sorted(
            self.image_dir.glob(IMAGE_SETTINGS["filename_pattern"]),
            key=lambda p: p.stat().st_mtime
        )
        self.image_paths = image_paths[::-1] if reversed else image_paths
        self.current_index = 0
        self.is_kitty = os.environ.get("TERM") == "xterm-kitty"
        self.ocr_cancelled = False

    def resize_image(self, img: Image.Image, max_width: int, max_height: int) -> Image.Image:
        width, height = img.size
        if width > max_width or height > max_height:
            if width / height > max_width / max_height:
                new_width = max_width
                new_height = int(height * (max_width / width))
            else:
                new_height = max_height
                new_width = int(width * (max_height / height))
            img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        return img

    def show_image(self):
        path = self.current_image()
        if self.is_kitty:
            self.kitty_show_image(path)
        else:
            print(" Install Kitty terminal for image previews: https://kitty.app")
        print(f"\n Current image: {path.name}")

    def kitty_show_image(self, path: Path):
        try:
            img = Image.open(path)
            img = self.resize_image(img, PREVIEW_SETTINGS["max_width"], PREVIEW_SETTINGS["max_height"])

            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            try:
                subprocess.run(["kitty", "+kitten", "icat", "--stdin"], input=img_bytes, check=True)
            except FileNotFoundError:
                print("\n Error: kitty command not found. Make sure Kitty is installed and in your PATH.")
            except subprocess.CalledProcessError as e:
                print(f"\n Error displaying image with kitty icat: {e}")

        except Exception as e:
            print(f"\n Image display error: {str(e)}")
            self.fallback_to_icat(path)

    def fallback_to_icat(self, path: Path):
        try:
            subprocess.run(["kitty", "icat", str(path)], check=True)
        except FileNotFoundError:
            print("\n Error: kitty command not found. Make sure Kitty is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            print(f"\n Error displaying image with kitty icat: {e}")

    def current_image(self) -> Path:
        return self.image_paths[self.current_index]

    def next_image(self):
        self.current_index =(self.current_index + 1)% len(self.image_paths) 
        #self.refresh_display()

    def prev_image(self):
        self.current_index =(self.current_index - 1)% len(self.image_paths)
        #self.refresh_display()

    def refresh_display(self):
        sys.stdout.write("\x1b[2J\x1b[H")
        self.show_image()

    async def stream_ocr(self, is_running: asyncio.Event) -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient() as client:
                image_path = self.current_image()
                img = Image.open(image_path)
                img = self.resize_image(img, OCR_SETTINGS["max_width"], OCR_SETTINGS["max_height"])

                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()
                image_base64 = base64.b64encode(img_bytes).decode()

                data = {
                    "model": OLLAMA_SETTINGS["vision_model"],
                    "prompt": "Extract text verbatim from this image. OCR only, no commentary.",
                    "images": [image_base64],
                    "stream": True
                }

                try:
                    async with client.stream(
                        "POST",
                        OLLAMA_SETTINGS["base_url"],
                        json=data,
                        timeout=None  # Allow indefinite streaming, cancellation will handle termination
                    ) as response:
                        if response.status_code != 200:
                            yield f"\n OCR API Error: HTTP Status Code {response.status_code}"
                            try:
                                error_content = await response.read()
                                yield f"\n Error Content: {error_content.decode()}"
                            except Exception as e:
                                yield f"\n Error reading error content: {str(e)}"
                            return

                        async for chunk in response.aiter_lines():
                            if self.ocr_cancelled:
                                yield "\n Aborted"
                                return
                            if chunk:
                                try:
                                    yield json.loads(chunk).get("response", "")
                                    await asyncio.sleep(0.01)  # Small delay to allow for cancellation
                                except json.JSONDecodeError as e:
                                    yield f"\n Error decoding JSON chunk: {str(e)}"
                                    yield f"\n Chunk data: {chunk}"
                except httpx.NetworkError as e:
                    yield f"\n Network error connecting to Ollama: {str(e)}"
                except httpx.TimeoutException as e:
                    yield f"\n Timeout error connecting to Ollama: {str(e)}"

        except Exception as e:
            yield f"\n General error in stream_ocr: {str(e)}"
        finally:
            is_running.clear()
            self.ocr_cancelled = False


class InputHandler:
    def __init__(self):
        self.session = PromptSession()
        self.main_kb, self.command_kb = self.create_keybindings()

    def create_keybindings(self) -> Tuple[KeyBindings, KeyBindings]:
        # Main keybindings with quick commands
        main_kb = KeyBindings()
        quick_commands = {'n': '/next', 'p': '/prev', 'o': '/ocr', 'q': '/quit'}
        for key, cmd in quick_commands.items():
            @main_kb.add(key)
            def _(event, cmd=cmd):
                event.app.exit(result=cmd)

        @main_kb.add('/')
        def _(event):
            event.app.exit(result='/')

        # Command keybindings - use default Emacs bindings
        command_kb = KeyBindings()

        # Add just the Enter and C-c bindings, others come from default Emacs mode
        @command_kb.add('enter')
        def _(event):
            event.app.exit(result=event.app.current_buffer.text)

        @command_kb.add('c-c')
        def _(event):
            event.app.exit(result=None)

        return main_kb, command_kb

    async def get_command(self):
        try:
            # Main prompt with quick commands
            cmd = await self.session.prompt_async(
                "> ",
                key_bindings=self.main_kb,
                enable_history_search=True,
                complete_while_typing=False,
            )

            if cmd == '/':
                # Command prompt with full readline capabilities
                full_cmd = await self.session.prompt_async(
                    "/",
                    key_bindings=self.command_kb,
                    # Enable Emacs mode for readline bindings
                    editing_mode=EditingMode.EMACS,
                    enable_history_search=True,
                    complete_while_typing=False,
                    # Add some common prompt-toolkit features
                    mouse_support=True,
                    wrap_lines=True,
                    multiline=False,
                )
                if full_cmd is None:
                    return None
                return f"/{full_cmd}"
            return cmd
        except asyncio.CancelledError:
            return None


async def main(image_dir: Path = None):
    print("Entering main function...")
    explorer = ImageExplorer(image_dir, reversed=True)
    print(f"Image directory: {explorer.image_dir}")
    print(f"Number of images found: {len(explorer.image_paths)}")

    if not explorer.image_paths:
        print(f"❌ No images found matching '{IMAGE_SETTINGS['filename_pattern']}'")
        print(f"   in directory: {explorer.image_dir}")
        return

    print("\x1b[2J\x1b[H Image OCR Explorer - Use n/p/o or /commands")
    if not explorer.is_kitty:
        print(" Install Kitty terminal: https://kitty.app")
    explorer.show_image()

    handler = InputHandler()
    ocr_running = asyncio.Event()
    ocr_task = None

    while True:
        try:
            cmd = await handler.get_command()
            if not cmd:
                continue

            print(f"\n→ {cmd}")

            if cmd == '/next':
                explorer.next_image()
                explorer.show_image()
            elif cmd == '/prev':
                explorer.prev_image()
                explorer.show_image()
            elif cmd == '/ocr':
                if not ocr_running.is_set():
                    ocr_running.set()
                    explorer.ocr_cancelled = False
                    print("\n OCR Results:")

                    async def process_ocr():
                        try:
                            async for token in explorer.stream_ocr(ocr_running):
                                print(token, end="", flush=True)
                                await asyncio.sleep(0.01)
                        except asyncio.CancelledError:
                            print("\n OCR Task Cancelled.")
                        except Exception as e:
                            print(f"\n Error during OCR processing: {e}")
                        finally:
                            print()
                            ocr_running.clear()

                    ocr_task = asyncio.create_task(process_ocr())
                    try:
                        await ocr_task
                    except asyncio.CancelledError:
                        print("\n OCR Aborted by User.")
                    ocr_task = None
                elif cmd == '/quit':
                    print("\n Exiting application.")
                    break
                else:
                    print("\n OCR is already in progress.")
            elif cmd == '/quit':
                print("\n Exiting application.")
                break
            else:
                print(f"❌ Unknown command: {cmd}")
                print("Available: /next, /prev, /ocr, /quit (or q)")

        except KeyboardInterrupt:
            print("\n Keyboard Interrupt Caught in Main Loop.")
            if ocr_running.is_set() and ocr_task is not None:
                explorer.ocr_cancelled = True
                print("\n Requesting OCR abortion...")
                if ocr_task and not ocr_task.done():
                    await ocr_task
                ocr_running.clear()
                ocr_task = None
            else:
                print("\n Operation cancelled")
                explorer.refresh_display()
        except Exception as e:
            print(f"\n Error: {str(e)}")

if __name__ == "__main__":
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if image_dir and not image_dir.is_dir():
        print(f"❌ Invalid directory: {image_dir}")
        sys.exit(1)
    asyncio.run(main(image_dir))

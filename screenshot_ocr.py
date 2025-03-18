#!/usr/bin/env python3
import os
import sys
import base64
import asyncio
import json
import signal
import subprocess
from pathlib import Path
from typing import List, AsyncGenerator, Tuple, Optional, Dict
import yaml

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.shortcuts import radiolist_dialog

from PIL import Image
import io

# ======================
# CONFIGURATION HANDLING
# ======================
DEFAULT_CONFIG = {
    "image_settings": {
        "default_directory": "~/Pictures/Screenshots",
        "filename_pattern": "Screenshot*.png"
    },
    "ollama_settings": {
        "base_url": "http://localhost:11434/api/generate",
        "models": ["llama3.2-vision"]
    },
    "preview_settings": {
        "max_width": 800,
        "max_height": 600
    },
    "ocr_settings": {
        "max_width": 1200,
        "max_height": 1200
    }
}

def load_config() -> Dict:
    """Load configuration from YAML files with fallback to defaults"""
    config_paths = [
        Path("screenshot_ocr.yaml"),
        Path.home() / ".config" / "screen-ocr" / "config.yaml"
    ]
    
    for path in config_paths:
        expanded_path = path.expanduser()
        if expanded_path.exists():
            try:
                with open(expanded_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    return {**DEFAULT_CONFIG, **config}  # Merge with defaults
            except Exception as e:
                print(f"Error loading config {path}: {str(e)}")
                return DEFAULT_CONFIG
    return DEFAULT_CONFIG

# ==================
# IMAGE EXPLORER
# ==================
class ImageExplorer:
    def __init__(self, config: Dict, image_dir: Path = None, reversed: bool = False):
        self.config = config
        self._init_image_dir(image_dir)
        self._init_models()
        self.current_index = 0
        self.is_kitty = os.environ.get("TERM") == "xterm-kitty"
        self.ocr_cancelled = False

    def _init_image_dir(self, image_dir: Optional[Path]):
        """Initialize image directory and paths"""
        default_dir = Path(self.config["image_settings"]["default_directory"]).expanduser()
        if not default_dir.exists():
            default_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_dir = image_dir or default_dir
        pattern = self.config["image_settings"]["filename_pattern"]
        
        self.image_paths = sorted(
            self.image_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime
        )
        if reversed:
            self.image_paths = self.image_paths[::-1]

    def _init_models(self):
        """Initialize model configuration"""
        ollama_config = self.config.get("ollama_settings", {})
        self.available_models = ollama_config.get("models", [])
        self.current_model = (
            self.available_models[0] 
            if self.available_models 
            else "llama3.2-vision"
        )

    def set_model(self, model_name: str):
        """Set current OCR model (no validation)"""
        self.current_model = model_name
        return True

    def resize_image(self, img: Image.Image, setting_key: str) -> Image.Image:
        """Resize image according to config settings"""
        settings = self.config.get(setting_key, {})
        max_w = settings.get("max_width", 800)
        max_h = settings.get("max_height", 600)
        
        width, height = img.size
        if width <= max_w and height <= max_h:
            return img
            
        ratio = min(max_w/width, max_h/height)
        new_size = (int(width*ratio), int(height*ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def _kitty_show_image(self, path: Path):
        """Display image using Kitty terminal's icat"""
        try:
            img = Image.open(path)
            img = self.resize_image(img, "preview_settings")

            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_buffer.flush()
            img_bytes = img_buffer.getvalue()

            try:
                subprocess.run(
                    ["kitty", "+kitten", "icat", "--align=left", "--stdin" ],
                    input=img_bytes,
                    check=True,
                )
            except FileNotFoundError:
                print("\n Error: kitty command not found. Make sure Kitty is installed and in your PATH.")
            except subprocess.CalledProcessError as e:
                print(f"\n Error displaying image with kitty icat: {e}")

        except Exception as e:
            print(f"\n Image processing error: {str(e)}")

    def show_image(self):
        """Display current image preview"""
        path = self.current_image()
        if self.is_kitty:
            print("\x1b[2J\x1b[H", end="", flush=True)  # Clear screen before showing image
            self._kitty_show_image(path)
        print(f"\n Current image: {path.name}")

    def current_image(self) -> Path:
        return self.image_paths[self.current_index]

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.show_image()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.show_image()

    async def stream_ocr(self, is_running: asyncio.Event) -> AsyncGenerator[str, None]:
        """Stream OCR results from Ollama API"""
        try:
            async with httpx.AsyncClient() as client:
                img_path = self.current_image()
                img = Image.open(img_path)
                img = self.resize_image(img, "ocr_settings")
                
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                b64_image = base64.b64encode(buffer.getvalue()).decode()
                
                data = {
                    "model": self.current_model,
                    "prompt": "Extract text verbatim from this image. OCR only, no commentary.",
                    "images": [b64_image],
                    "stream": True
                }
                
                async with client.stream(
                    "POST",
                    self.config["ollama_settings"].get("base_url", DEFAULT_CONFIG["ollama_settings"]["base_url"]),
                    json=data,
                    timeout=30
                ) as response:
                    if response.status_code != 200:
                        yield f"\n OCR Error ({response.status_code}): {await response.atext()}"
                        return
                        
                    async for chunk in response.aiter_lines():
                        if self.ocr_cancelled:
                            yield "\n OCR aborted"
                            return
                        try:
                            yield json.loads(chunk).get("response", "")
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"\n OCR Error: {str(e)}"
        finally:
            is_running.clear()
            self.ocr_cancelled = False

# ==================
# INPUT HANDLER
# ==================
class InputHandler:
    def __init__(self):
        self.session = PromptSession()
        self.main_kb, self.command_kb = self._create_keybindings()

    def _create_keybindings(self) -> Tuple[KeyBindings, KeyBindings]:
        """Create keybindings for prompt sessions"""
        main_kb = KeyBindings()
        quick_actions = {
            'n': '/next',
            'p': '/prev',
            'o': '/ocr',
            'm': '/model',
            'q': '/quit'
        }
        
        for key, cmd in quick_actions.items():
            @main_kb.add(key)
            def _(event, cmd=cmd):
                event.app.exit(result=cmd)

        @main_kb.add('/')
        def _(event):
            event.app.exit(result='/')

        command_kb = KeyBindings()
        @command_kb.add('enter')
        def _(event):
            event.app.exit(result=event.app.current_buffer.text)
        @command_kb.add('c-c')
        def _(event):
            event.app.exit(result=None)
            
        return main_kb, command_kb

    async def get_command(self) -> Optional[str]:
        """Get user command with support for quick keys and full input"""
        try:
            cmd = await self.session.prompt_async(
                "> ",
                key_bindings=self.main_kb,
                enable_history_search=True
            )
            
            if cmd == '/':
                full_cmd = await self.session.prompt_async(
                    "/",
                    key_bindings=self.command_kb,
                    editing_mode=EditingMode.EMACS
                )
                return f"/{full_cmd}" if full_cmd else None
                
            return cmd
        except asyncio.CancelledError:
            return None

# ==================
# MAIN APPLICATION
# ==================
async def main(image_dir: Path = None):
    config = load_config()
    explorer = ImageExplorer(config, image_dir, reversed=True)
    
    if not explorer.image_paths:
        print(f"❌ No images found matching '{config['image_settings']['filename_pattern']}'")
        return

    print("\x1b[2J\x1b[H Screenshot OCR Explorer")
    explorer.show_image()
    
    handler = InputHandler()
    ocr_task = None
    ocr_running = asyncio.Event()

    while True:
        try:
            cmd = await handler.get_command()
            if not cmd:
                continue
                
            print(f"\n→ {cmd}")

            if cmd == '/next':
                explorer.next_image()
            elif cmd == '/prev':
                explorer.prev_image()
            elif cmd == '/ocr':
                if not ocr_running.is_set():
                    ocr_running.set()
                    print("\n OCR Results:")
                    ocr_task = asyncio.create_task(_run_ocr(explorer, ocr_running))
                    await ocr_task
                else:
                    print("\n OCR already in progress")
            elif cmd.startswith('/model'):
                await handle_model_command(cmd, explorer)
            elif cmd == '/quit':
                break
            else:
                print(f" Unknown command: {cmd}")
                print("Available: /next(n), /prev(p), /ocr(o), /model(m), /quit(q)")
                
        except KeyboardInterrupt:
            if ocr_running.is_set():
                explorer.ocr_cancelled = True
                print("\n Cancelling OCR...")
                await ocr_task
            ocr_running.clear()

async def _run_ocr(explorer: ImageExplorer, ocr_running: asyncio.Event):
    """Handle OCR streaming output"""
    try:
        async for token in explorer.stream_ocr(ocr_running):
            print(token, end="", flush=True)
    finally:
        print()
        ocr_running.clear()

async def handle_model_command(cmd: str, explorer: ImageExplorer):
    """Handle model selection commands"""
    if ' ' in cmd:
        model_name = cmd.split(' ', 1)[1]
        explorer.set_model(model_name)
        print(f" Model set to: {model_name}")
    else:
        if explorer.available_models:
            selection = await radiolist_dialog(
                title="Select Model",
                text="Recommended models from config:",
                values=[(m, m) for m in explorer.available_models],
            ).run_async()
            
            if selection:
                explorer.set_model(selection)
                print(f" Model changed to: {selection}")
        else:
            print(" No recommended models in config. Use '/model <name>' to set manually.")

if __name__ == "__main__":
    config = load_config()
    image_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    asyncio.run(main(image_dir))

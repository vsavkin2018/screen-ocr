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
import copy

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.enums import EditingMode
#from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.completion import WordCompleter

from PIL import Image
import io
from abc import ABC, abstractmethod

DEFAULT_CONFIG = {
    "image_settings": {
        "default_directory": "~/Pictures/Screenshots",
        "filename_pattern": "Screenshot*.png"
    },
    "ollama": {  
        "base_url": "http://localhost:11434/api/generate",
        "models": ["llama3.2-vision"],
        "max_width": 1200,  
        "max_height": 1200,
        "current_prompt": None, # this will be changes in runtime
        "prompt_list": [ # will be overrided by "prompts" if exists
            ("simple", "Extract text verbatim from this image. OCR only, no commentary."),
            ("generic", """Act as an OCR assistant. Analyze the provided image and:
                1. Recognize all visible text in the image as accurately as possible.
                2. Maintain the original structure and formatting of the text.
                3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.

                Provide only the transcription without any additional comments.""" )
        ],
        
               
    },
    "preview_settings": {
        "max_width": 800,
        "max_height": 600
    }
}

def deep_merge(dst: Dict, src: Dict):
    for key, value in src.items():
        if isinstance(value, dict) and key in dst:
            deep_merge(dst[key], value)
        else:
            dst[key] = value

def load_config() -> tuple[dict, list[str]]:
    """Load configuration and return (config, warnings)"""
    config = copy.deepcopy(DEFAULT_CONFIG)
    warnings = []

    config_paths = [
        Path.home() / ".config" / "screen-ocr" / "config.yaml",
        Path("screenshot_ocr.yaml"),
    ]

    for path in config_paths:
        expanded_path = path.expanduser()
        if expanded_path.exists():
            try:
                with open(expanded_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}

                    if 'ocr_settings' in user_config:
                        warnings.append(
                            f"⚠️  Config warning: 'ocr_settings' section in {path} is deprecated.\n"
                            "   Please move 'max_width'/'max_height' under 'ollama' section"
                        )

                    deep_merge(config, user_config)

            except Exception as e:
                warnings.append(f"Error loading config {path}: {str(e)}")

    return config, warnings

# ==================
# IMAGE PROCESSING
# ==================
def resize_image(img: Image.Image, max_width: Optional[int], max_height: Optional[int]) -> Image.Image:
    """Universal image resizing maintaining aspect ratio"""
    if max_width is None and max_height is None:
        return img

    width, height = img.size
    effective_max_w = max_width or width  # Treat None as no constraint
    effective_max_h = max_height or height

    if width <= effective_max_w and height <= effective_max_h:
        return img

    ratio = min(effective_max_w/width, effective_max_h/height)
    new_size = (int(width*ratio), int(height*ratio))
    return img.resize(new_size, Image.Resampling.LANCZOS)

# ======================================================================
# CTRL+C HANDLING STRATEGY FOR OCR OPERATIONS (Multi-Engine Version)
# ======================================================================

# ======================================================================
# CRITICAL INTERRUPT HANDLING (DO NOT MODIFY WITHOUT TESTING)
# ======================================================================
#
# Implementation Strategy for Ctrl+C/Ctrl+D:
#
# 1. Ctrl+C (KeyboardInterrupt) Handling:
#    - During OCR Processing:
#      a) Main app calls current_engine.cancel()
#      b) Engine performs implementation-specific abort:
#         - HTTP engines: Close connection
#         - Subprocess engines: Send SIGTERM
#         - Threaded engines: Set stop flags
#      c) Engine's generator yields abort message
#      d) Automatically returns to command prompt
#    - At Idle State:
#      a) Discard current input (if any)
#      b) Show new command prompt
#
# 2. Ctrl+D (EOF) Handling:
#    a) Clean exit with "Exiting..." confirmation
#    b) Cancel current engine if active
#    c) Safely closes event loop
#
# Key Components:
# - BaseEngine.cancel(): Unified cancellation interface
# - Engine-owned cleanup: Each implementation manages its resources
# - Main app coordination: Tracks active engine instance
#
# Error Prevention:
# 1. Single cancellation entry point (main app → engine.cancel())
# 2. No direct task cancellation - engine-specific cleanup
# 3. Connection/process cleanup owned by engine implementations
# 4. Timeout enforcement per engine requirements
#
# Critical Implementation Details:
# 1. Engine implementations MUST:
#    - Implement cancel() to stop active operations atomically
#    - Ensure generator termination after cancellation
#    - Yield exactly ONE abort message when cancelled
#    - Handle cleanup of network/subprocess resources
#
# 2. The main app MUST:
#    - Maintain reference to current_engine during operations
#    - Call cancel() on KeyboardInterrupt
#    - Never manage engine-specific resources directly
#    - Reset current_engine reference after completion
#
# 3. EOF handling MUST:
#    - Break main loop immediately
#    - Call cancel() if engine is active
#
# Edge Case Guarantees:
# - Ctrl+C during HTTP streaming: Engine closes connection
# - Ctrl+C during subprocess execution: Engine sends SIGTERM
# - Rapid double Ctrl+C: Treated as single abort
# - Ctrl+D during OCR: Full exit with engine cleanup
# - Mixed engine types: Uniform cancellation behavior
#
# WARNING: This interrupt handling is engine-dependent. Modifications MUST:
# - Preserve BaseEngine interface contract
# - Maintain engine-owned resource cleanup
# - Test all scenarios per engine type:
#   1. HTTP-based engines (Ollama)
#   2. Subprocess-based engines (Tesseract)
#   3. Thread-pool engines
#   4. Mixed interrupt sequences across types
# ======================================================================


# ==================
# OCR ENGINES
# ==================
class BaseEngine(ABC):
    @abstractmethod
    async def stream_ocr(self, image: Image.Image) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    def cancel(self):
        pass

    def get_max_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (max_width, max_height) for this engine. None means no limit."""
        return (None, None)  # Default: no resizing

    def prepare_image(self, img: Image.Image) -> Image.Image:
        """Resize image according to engine requirements"""
        max_width, max_height = self.get_max_dimensions()
        return resize_image(img, max_width, max_height)

class DummyEngine(BaseEngine):
    def __init__(self):
        pass
    def cancel(self):
        pass
    async def stream_ocr(self, image: Image.Image) -> AsyncGenerator[str, None]:
        yield("Nothing to see here!")

class OllamaEngine(BaseEngine):
    def __init__(self, config: dict):
        self.config = config["ollama"]
        self._active_request: Optional[httpx.Response] = None
        self._cancelled = False

    def get_max_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        return self.config.get("max_width", 1200), self.config.get("max_height", 1200)

    def cancel(self):
        """Cancel current OCR operation"""
        self._cancelled = True
        if self._active_request:
            # Close the HTTP connection immediately
            asyncio.create_task(self._active_request.aclose())

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
                    "model": self.config.get("model", "llama3.2-vision"),
                    "prompt": self.config["current_prompt"],
                    "images": [b64_image],
                    "stream": True
                }

                try:
                    self._active_request = client.stream(
                        "POST",
                        self.config.get("base_url", DEFAULT_CONFIG["ollama"]["base_url"]),
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

# Used to select and create engines
ENGINE_LIST = [
        ("ollama", lambda conf : OllamaEngine(conf)),
        ("dummy", lambda conf : DummyEngine())
]

# ==================
# IMAGE EXPLORER
# ==================
class ImageExplorer:
    def __init__(self, config: Dict, image_dir: Path = None):
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
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def _init_models(self):
        """Initialize model configuration"""
        ollama_config = self.config.get("ollama", {})
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

    def get_preview_image(self, path: Path) -> Image.Image:
        """Get resized image for preview"""
        img = Image.open(path)
        settings = self.config["preview_settings"]
        return resize_image(
            img,
            settings.get("max_width"),
            settings.get("max_height")
        )

    def _kitty_show_image(self, path: Path):
        """Display image preview using Kitty terminal's icat"""
        try:
            img = self.get_preview_image(path)

            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_buffer.flush()
            img_bytes = img_buffer.getvalue()

            try:
                subprocess.run(
                    ["kitty", "+kitten", "icat", "--align=left", "--stdin=yes" ],
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
    
    def create_engine(self, engine_type: str = "ollama") -> BaseEngine:
        """Factory method for OCR engines"""
        for name, create_func in ENGINE_LIST:
            if engine_type == name:
                return create_func(self.config)
        raise ValueError(f"Unknown engine type: {engine_type}")


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
            'e': '/engine',
            'P': '/prompt',
            'q': '/quit',
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
    config, warnings = load_config()
    explorer = ImageExplorer(config, image_dir)
    current_engine: Optional[BaseEngine] = None  # Track active engine
    engine_type = ENGINE_LIST[0][0]
    
    print("\x1b[2J\x1b[H Screenshot OCR Explorer")
    if warnings:
        print("\n".join(warnings))
    if not explorer.image_paths:
        print(f"❌ No images found matching '{config['image_settings']['filename_pattern']}'")
        return

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
                if not current_engine:
                    current_engine = explorer.create_engine(engine_type)
                if not ocr_running.is_set():
                    ocr_running.set()
                    print("\n OCR Results:")
                    ocr_task = asyncio.create_task(_run_ocr(current_engine, explorer, ocr_running))
                    try:
                        await ocr_task
                    except asyncio.CancelledError:
                        print("\n OCR aborted")
                else:
                    print("\n OCR already in progress")
            elif cmd.startswith('/model'):
                await handle_model_command(cmd, explorer)
            elif cmd.startswith('/prompt'):
                await handle_prompt_command(cmd, config)
            elif cmd.startswith('/engine'):
                engine_type = await handle_engine_command(cmd)
                if current_engine:
                    current_engine.cancel()
                current_engine = None
            elif cmd == '/quit':
                break
            else:
                print(f" Unknown command: {cmd}")
                print("Available: /next(n), /prev(p), /ocr(o), /model(m), /quit(q)")

        except KeyboardInterrupt:
            if ocr_running.is_set():
                explorer.ocr_cancelled = True
                print() # some workaround probably
            if current_engine:
                current_engine.cancel()
            ocr_running.clear()
            current_engine = None
        except EOFError:
            if current_engine:
                current_engine.cancel()
            print("quit")
            break

    # Cancel any remaining tasks when exiting
    if ocr_task and not ocr_task.done():
        ocr_task.cancel()
        try:
            await ocr_task
        except asyncio.CancelledError:
            pass

async def _run_ocr(engine: BaseEngine, explorer: ImageExplorer, ocr_running: asyncio.Event):
    """Handle OCR streaming output"""
    try:
        img_path = explorer.current_image()
        img = Image.open(img_path)
        img = engine.prepare_image(img)

        async for token in engine.stream_ocr(img):
            print(token, end="", flush=True)
    finally:
        print()
        ocr_running.clear()

async def make_multiple_choice(L: list, prompt: str) -> Optional[str]:
    """
    Let user choose from the list.
    Elements of the list are either strings or (name, description) tuples

    Returns a string (name) or None
    """
    max_len = 70
    LL = []
    for elem in L:
        if isinstance(elem, tuple) or isinstance(elem, list):
            name, desc = elem
            desc = ' '.join(desc.split()) # replace multiple whitespaces by single space
            printable = f"{name}: {desc}"
            if len(printable)>max_len:
                printable = printable[:max_len-3] + "..."
            LL.append((name, printable))
        else:
            LL.append((elem, elem))
    #print ("debug:", L, LL)
    for i,(_,printable) in enumerate(LL, 1):
        print(f" {i}. {printable}")

    completer = WordCompleter([name for name,_ in LL])
    session = PromptSession()

    selection = None
    try:
        selection = await session.prompt_async(
                prompt,
                completer=completer,
                complete_while_typing=True,
            )

        # Try to convert to index
        if selection.isdigit():
            try:
                index = int(selection) -1
                if index<0:
                    raise IndexError()
                selection = LL[index][0]
            except ValueError:
                pass
            except IndexError:
                print("No such index:",selection)
                return None
    except (KeyboardInterrupt, EOFError):
        print("\n cancelled")
    return selection


async def handle_model_command(cmd: str, explorer: ImageExplorer):
    """Handle model selection commands"""
    if ' ' in cmd:
        model_name = cmd.split(' ', 1)[1]
        explorer.set_model(model_name)
        print(f" Model set to: {model_name}")
    else:
        if explorer.available_models:
            print("Recommendef models from config:")
            selection = await make_multiple_choice(
                explorer.available_models,
                "Select model: "
            )
            
            if selection:
                explorer.set_model(selection)
                print(f" Model changed to: {selection}")
        else:
            print(" No recommended models in config. Use '/model <name>' to set manually.")

async def handle_prompt_command(cmd: str, config: Dict):
    """Handle prompt selection commands"""
    pr = None
    if ' ' in cmd:
        pr = cmd.split(' ', 1)[1]
    else:
        print("Suggested prompts:")
        PL = config["ollama"]["prompt_list"]
        prname = await make_multiple_choice(
                PL,
                "Select prompt: "
            )
        if not prname:
              prname = PL[0][0]

        for n,p in PL:
            if n==prname:
                pr = p
    if not pr:
        print("Prompt not changed")
    else:
        print(f" New prompt:\n{pr}\n")
        config["ollama"]["current_prompt"] = pr

async def handle_engine_command(cmd: str) -> str:
    """Handle engine selection commands"""
    if ' ' in cmd:
        name = cmd.split(' ', 1)[1]
    else:
        print("Available engines:")
        name = await make_multiple_choice(
                [n for n,_ in ENGINE_LIST],
                "Select engine: "
            )
        if not name:
              name = ENGINE_LIST[0][0]
    print(f" Engine set to: {name}")
    return name

def real_main():
    config = load_config()
    image_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    asyncio.run(main(image_dir))

if __name__ == "__main__":
    real_main()

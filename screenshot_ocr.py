#!/usr/bin/env python3
import os
import sys
import tty
import termios
import base64
import asyncio
import json
import signal
import subprocess
from pathlib import Path
from typing import List, AsyncGenerator, Tuple, Optional, Dict
import yaml
import copy

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

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
        "model": None, # this will be changed at runtime
        "current_prompt": None, # this will be changed at runtime
        "prompt_list": [ # will be overrided by "prompts" if exists
            ("simple", "Extract text verbatim from this image. OCR only, no commentary."),
            ("generic", """Act as an OCR assistant. Analyze the provided image and:
                1. Recognize all visible text in the image as accurately as possible.
                2. Maintain the original structure and formatting of the text.
                3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.

                Provide only the transcription without any additional comments.""" )
        ],
        "chat_prompt": """You're an assistant working with OCR results. 
        Current OCR context: {ocr_text}
        User question: {question}
        Please provide your answer using both image and text:""",
               
    },
    "tesseract": {
        "psm": 3,
        "oem": 1,
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

    # Initialize model configuration
    ollama_config = config.get("ollama", {})
    available_models = ollama_config.get("models", [])
    current_model = (
        available_models[0] 
        if available_models 
        else "llama3.2-vision" )
    ollama_config["model"] = current_model
    # TODO: Parsing prompt dict-list

    return config, warnings


# ======================================================================
# CTRL+C HANDLING STRATEGY FOR OCR OPERATIONS (Multi-Engine Version)
# ======================================================================

# When in command prompt, SIGINT is not generated, Ctrl+C is handled by prompt_tookit by key binding.
# When waiting for completion of a task, _sigint is responsible for cancelling the current task.
# _sigint access global variables: ocr_task and handler.input_task, cancelling that which is not None.
# We have to restore the signal handler after call to prompt_async (_prompt_wrapper does it).
# Engine must implement cancel() method to close all external resources and shut down any processes.


# ==================
# OCR ENGINES
# ==================
from ocr_base import *
from ollama_engine import OllamaEngine
from tesseract_engine import TesseractEngine

# Used to select and create engines
ENGINE_LIST = [
        ("tesseract", lambda conf: TesseractEngine(conf)),
        ("ollama", lambda conf : OllamaEngine(conf)),
        ("dummy", lambda conf : DummyEngine())
]

def create_engine(engine_type: str, config: Dict) -> BaseEngine:
    """Factory method for OCR engines"""
    for name, create_func in ENGINE_LIST:
        if engine_type == name:
            return create_func(config)
    raise ValueError(f"Unknown engine type: {engine_type}")

# IMAGE EXPLORER
from explore_images import ImageExplorer    

# ==================
# INPUT HANDLER
# ==================
class InputHandler:
    def __init__(self):
        self.history = InMemoryHistory()
        self.session = PromptSession(history=self.history)
        self.chat_session = None
        self.main_kb, self.command_kb, self.chat_kb = self._create_keybindings()
        self.input_task = None

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
            'c': '/chat',
            '.': '/refresh', 'R': '/refresh',
            'q': '/quit',
        }
        
        for key, cmd in quick_actions.items():
            @main_kb.add(key)
            def _(event, cmd=cmd):
                event.app.exit(result=cmd)

        @main_kb.add('/')
        def _(event):
            event.app.exit(result='/')

        @main_kb.add('c-c')
        def _(event):
            event.app.exit(result=None)

        command_kb = KeyBindings()
        @command_kb.add('c-c')
        def _(event):
            event.app.exit(result=None)

        chat_kb = KeyBindings()
        @chat_kb.add('c-c')
        def _(event):
            event.app.exit(result='/')
            
        return main_kb, command_kb, chat_kb

    async def _prompt_wrapper(self, session: PromptSession, prompt: str, **kwargs) -> Optional[str]:
        res = None
        fd = sys.__stdin__.fileno()
        tty_attrs = termios.tcgetattr(fd)

        try:
            self.input_task = asyncio.create_task( session.prompt_async(
                prompt, handle_sigint=False, **kwargs
            ) )
            res = await self.input_task

        except asyncio.CancelledError:
            print(f"\n{sys._getframe(1).f_code.co_name}: _prompt_wrapper got CancelledError")
            res = None
        finally:
            self.input_task = None
            termios.tcsetattr(fd, termios.TCSAFLUSH, tty_attrs)
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, _sigint)

        return res

    async def get_command(self) -> Optional[str]:
        """Get user command with support for quick keys and full input"""

        cmd = await self._prompt_wrapper(self.session,
                "> ",
                key_bindings=self.main_kb,
                enable_history_search=False,
            ) 
        if cmd == '/':
            full_cmd = await self._prompt_wrapper ( self.session,
                    "", default="/",
                    key_bindings=self.command_kb,
                    editing_mode=EditingMode.EMACS,
                    multiline=False,
                    enable_history_search=True,
                ) 
            cmd = full_cmd if full_cmd else None
                
        return cmd

    async def get_chat_input(self) -> Optional[str]:
        """Multiline input with natural interrupt handling"""
        session = self.chat_session
        if not session:
            session = self.chat_session = PromptSession()
        def cont(width, line_number, is_soft_wrap):
            return "" if is_soft_wrap else "..."+" "*max(0, width-3)
        try:
            line = await self._prompt_wrapper(session,
                "Chat> " ,
                key_bindings=self.chat_kb,
                multiline=True,
                prompt_continuation = cont,
                editing_mode=EditingMode.EMACS,
            )
            line = line.replace("//", "/", 1)
            return line
        except EOFError:
            return None

    async def make_multiple_choice(self, L: list, prompt: str) -> Optional[str]:
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

        kb = KeyBindings()
        @kb.add('c-c')
        def _(event):
            event.app.exit(result=None)

        selection = None
        try:
            selection = await self._prompt_wrapper(session,
                    prompt,
                    completer=completer,
                    key_bindings=kb,
                    complete_while_typing=True,
                )
            if not selection:
                return selection

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

class OCRContext:
    "Current context used in the main loop"
    def __init__(self):
        self.reset()

    def reset_chat(self):
        self.chat_context: Optional[dict] = None

    def reset(self):
        self.reset_chat()
        self.last_ocr_text: str = ""
        self.last_ocr_image: Optional[Image.Image] = None

# Ctrl+C handler
def _sigint():
    global ocr_task
    global handler

    #print("plonk!")
    if ocr_task:
        #print(repr(ocr_task))
        ocr_task.cancel()
    if handler and handler.input_task:
        #print(repr(handler.input_task))
        handler.input_task.cancel()

# ==================
# MAIN APPLICATION
# ==================
async def main(image_dir: Path = None, show_banner = True):
    config, warnings = load_config()
    current_engine: Optional[BaseEngine] = None  # Track active engine
    engine_type = ENGINE_LIST[0][0]
    explorer = None
    
    if show_banner:
        print("\x1b[2J\x1b[H Screenshot OCR Explorer")
    if warnings:
        print("\n".join(warnings))

    def init_explorer():
        nonlocal explorer
        explorer = ImageExplorer(config, image_dir)
        if not explorer.image_sources:
            print(f"❌ No images found matching '{config['image_settings']['filename_pattern']}'")
            return
        explorer.show_image()

    init_explorer()
    global handler
    handler = InputHandler()
    global ocr_task
    ocr_task = None

    ctx = OCRContext()


    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _sigint)

    while True:

        if handler.input_task:
            # Should not happen
            print(f"\nMain Loop: input_task={repr(handler.input_task)}")
            handler.input_task.cancel()
            handler.input_task = None

        if ctx.chat_context:  # Chat mode
            try:
                # Validate image hasn't changed
                current_image = explorer.current_image().get_description()
                if ctx.chat_context['image_desc'] != current_image:
                    print("\n⚠️ Image changed! Perform new OCR first.")
                    ctx.reset_chat()
                    continue

                user_input = await handler.get_chat_input()
                if (user_input is None) or (user_input == "/"):
                    ctx.reset_chat()
                    print("\nExited chat mode")
                    continue

                print()  # Response separator
                if not current_engine:
                    current_engine = create_engine(engine_type, config)
                try:
                    if ocr_task and not ocr_task.done():
                        await ocr_task
                except Exception as e:
                    print(f"\n Error while terminating: {str(e)}")

                ocr_task = asyncio.create_task(_run_chat_question(
                    current_engine.stream_chat(user_input, ctx.chat_context),
		    current_engine))
                await ocr_task
                ocr_task = None

            except KeyboardInterrupt:
                if current_engine:
                    await current_engine.cancel()
                    current_engine = None
                ctx.reset_chat()
                print("\nChat interrupted")
                continue

            continue # Chat mode

        try:
            cmd = await handler.get_command()
            if not cmd:
                continue
                
            print(f"→ {cmd}")

            if cmd == '/next':
                explorer.next_image()
                ctx.reset()
            elif cmd == '/prev':
                explorer.prev_image()
                ctx.reset()
            elif cmd == '/ocr':
                if not current_engine:
                    current_engine = create_engine(engine_type, config)
                ctx.reset()

                try:
                    if ocr_task and not ocr_task.done():
                        await ocr_task
                except Exception as e:
                    print(f"\n Error while terminating: {str(e)}")

                print("\n OCR Results:")
                ocr_task = asyncio.create_task(_run_ocr(current_engine, explorer, ctx))
                try:
                    await ocr_task
                    ocr_task = None
                except asyncio.CancelledError:
                    print("\n OCR aborted")
            elif cmd.startswith('/model'):
                await handle_model_command(cmd, config, handler)
            elif cmd.startswith('/prompt'):
                await handle_prompt_command(cmd, config, handler)
            elif cmd.startswith('/engine'):
                engine_type = await handle_engine_command(cmd, handler)
                if current_engine:
                    await current_engine.cancel()
                current_engine = None
            elif cmd == '/chat':
                if not ctx.last_ocr_text or not ctx.last_ocr_image:
                    print("No OCR context available. Perform OCR first")
                    continue

                # Store prepared image for chat sessions
                ctx.chat_context = {
                    'engine': engine_type,
                    'image_desc': explorer.current_image().get_description(),
                    'ocr_text': ctx.last_ocr_text,
                    'prepared_image': ctx.last_ocr_image
                }
                print(f"\nEntered chat mode (image: {ctx.chat_context['image_desc']})")
                print("Meta+Enter for enter, / or Ctrl+D for exit")
            elif cmd == '/refresh':
                ctx.reset()
                init_explorer()
            elif cmd == '/quit':
                break
            else:
                print(f" Unknown command: {cmd}")
                print("Available: /next(n), /prev(p), /ocr(o), /model(m), /quit(q)")

        except KeyboardInterrupt:
            explorer.ocr_cancelled = True
            print() # some workaround probably
            if current_engine:
                await current_engine.cancel()
            current_engine = None
        except EOFError:
            if current_engine:
                await current_engine.cancel()
                current_engine = None
            print("quit")
            break

    # Cancel any remaining tasks when exiting
    if ocr_task and not ocr_task.done():
        ocr_task.cancel()
        try:
            await ocr_task
        except asyncio.CancelledError:
            pass

    if current_engine:
        await current_engine.cancel()

async def _run_ocr(engine: BaseEngine, explorer: ImageExplorer, ocr_ctx: OCRContext):
    """Handle OCR streaming output"""
    buffer = io.StringIO()
    try:
        source = explorer.current_image()
        img = source.get_image()
        processed_img = ocr_ctx.last_ocr_image = engine.prepare_image(img)

        async for token in engine.stream_ocr(processed_img):
            print(token, end="", flush=True)
            buffer.write(token)

        ocr_ctx.last_ocr_text =  buffer.getvalue()

    except asyncio.CancelledError:
        print("\n OCR task cancelled")
        await engine.cancel()
    finally:
        print()

async def _run_chat_question(g, engine):
    """Prints chat output in its own task"""
    try:
        async for token in g:
            print(token, end="", flush=True)
    except asyncio.CancelledError:
        print("\n OCR task cancelled")
        await engine.cancel()
    finally:
        print()  # Ensure clean prompt


async def handle_model_command(cmd: str, config: Dict, ih: InputHandler):
    """Handle model selection commands"""
    model_name = None
    if ' ' in cmd:
        model_name = cmd.split(' ', 1)[1]
    else:
        available_models = config["ollama"].get("models", [])
        if available_models:
            print("Recommended models from config:")
            model_name = await ih.make_multiple_choice(
                available_models,
                "Select model: "
            )
        else:
            print(" No recommended models in config. Use '/model <name>' to set manually.")
    if model_name:
        config["ollama"]["model"] = model_name
        print(f" Model set to: {model_name}")

async def handle_prompt_command(cmd: str, config: Dict, ih: InputHandler):
    """Handle prompt selection commands"""
    pr = None
    if ' ' in cmd:
        pr = cmd.split(' ', 1)[1]
    else:
        print("Suggested prompts:")
        PL = config["ollama"]["prompt_list"]
        prname = await ih.make_multiple_choice(
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

async def handle_engine_command(cmd: str, ih: InputHandler) -> str:
    """Handle engine selection commands"""
    if ' ' in cmd:
        name = cmd.split(' ', 1)[1]
    else:
        print("Available engines:")
        name = await ih.make_multiple_choice(
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
    flag = False
    show_banner = True
    while not flag:
        try:
            save_show_banner = show_banner
            show_banner = False
            asyncio.run(main(image_dir, save_show_banner))
            flag = True
        except KeyboardInterrupt:
            print("\nspurious interrupt")
            pass


if __name__ == "__main__":
    real_main()

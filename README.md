# screen-ocr

This program is designed for recognizing text in screenshots sent by users, especially in Telegram for example.
Recognition is supported via tesseract and LLM through Ollama. There are plans to add other engines, such as easyocr and paddleocr.

## Installation

1.  **For working with Ollama, you need to install Ollama:** [https://github.com/ollama/ollama/blob/main/docs/linux.md](https://github.com/ollama/ollama/blob/main/docs/linux.md)

    There, simply download the archive and unpack it. You can run it directly from where it's unpacked, for example, `~/Build/ollama/bin/ollama serve`. This command should start and wait for requests on `http://localhost:11434`. Leave it running in a separate terminal or in some other way.

    Next, you need to download a vision LLM.

    If you have a GPU with 8GB of VRAM, you can download `gemma3:4b`. You can also download `llama3.2-vision` - it's larger and won't fit in such video memory, so it will work slower.
    You can search for models on the Ollama website by the word "vision", for example.

    Currently, the script has `llama3.2-vision` hardcoded as the default model, but you can choose another one in the config or at runtime.

    That is, to get started, you can download the model like this: `~/Build/ollama/bin/ollama pull llama3.2-vision` - it's about 10 GB or so, you need to wait and have enough space.

2.  **For working with tesseract, you need to install tesseract:**
    ```bash
    apt install tesseract-ocr-all # On Debian
    ```
    In the config file, I recommend setting at least the languages. Details are in the tesseract manual.

3.  **I recommend installing the kitty terminal, it's needed for the graphical preview of what is being recognized:**
    ```bash
    apt install kitty kitty-shell-integration kitty-terminfo
    ```
    Also, working with the clipboard requires external commands, xclip or wl-paste:
    ```bash
    apt install x11-apps # for X11
    apt install wl-clipboard # for Wayland
    ```

4.  **Create a Python virtual environment and install the libraries there:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -U pip setuptools  && pip install pyyaml httpx prompt_toolkit pillow pytesseract
    ```

5.  **Place the config file:**
    ```bash
    mkdir -p ~/.config/screen-ocr
    cp config.yaml ~/.config/screen-ocr/
    vi ~/.config/screen-ocr/config.yaml
    ```

6.  **You can run it directly from the terminal (kitty or another):** `python3 screenshot_ocr.py` - but you can assign the provided `start_ocr` script to a hotkey, which starts kitty with the OCR process there. First, you need to install the script inside the venv:
    ```bash
    pip install -e .
    # you can run ./start_ocr
    ```

## How to Use

By default, the program takes images of the type `~/Pictures/Screenshots/Screenshot*.png` - this can be configured. This is where Gnome3 under Debian or Ubuntu puts screenshots. That is, you can use Gnome tools to select an area on the screen, it will be saved to a file, and then you can run `start_ocr` by hotkey. However, the first in the list is the image from the clipboard, if it's there. So you can use any screenshot tool that can leave the result in the clipboard.

If you are not using a DE, you can take a screenshot with an external program, for example, install flameshot and call it by hotkey (PrtScr) with `flameshot gui` or `shutter -s`.

The program opens the latest screenshot. You can cycle through the screenshots using the `n` and `p` keys. OCR starts with the `o` key. It might take a while initially (LLMs work slowly, tesseract works faster), and then the text will appear. `Ctrl+C` can interrupt the process. You can select the text in the standard way with the mouse and copy it. `Ctrl+Shift+C` - kitty copies to the clipboard.

Press `q` at the end to exit.

There is a choice of engine with the `e` key.
For the ollama engine, there is a choice of model: with the `m` key from the list in the config. Or even `/model <name>`. You can use `/prompt <text>` to set a prompt.
Or use `Shift+P` to select a prompt from the list (default or in the config).

There is an option to clarify in chat mode. After recognition, a chat opens with the "c" key, you can enter a question.
You can enter several questions, but so far the context is not taken into account, only the OCR result and the last question.

Exit the chat with the command "/" or `Ctrl+D`.

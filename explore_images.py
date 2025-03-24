from typing import List, AsyncGenerator, Tuple, Optional, Dict
import os
from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod

# ==================
# IMAGE EXPLORER
# ==================
class ImageExplorer:
    def __init__(self, config: Dict, image_dir: Path = None):
        self.config = config
        self._init_image_dir(image_dir)
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

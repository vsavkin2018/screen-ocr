from typing import List, AsyncGenerator, Tuple, Optional, Dict
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageGrab
import hashlib
from abc import ABC, abstractmethod

from ocr_base import *

# ======================
# IMAGE SOURCE CLASSES
# ======================
class AbstractImageSource(ABC):
    @abstractmethod
    def get_image(self) -> Image.Image:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

class FileImageSource(AbstractImageSource):
    def __init__(self, path: Path):
        self.path = path
        self._image: Optional[Image.Image] = None

    def get_image(self) -> Image.Image:
        if not self._image:
            self._image = Image.open(self.path)
        return self._image

    def get_description(self) -> str:
        return f"file:{self.path.name}"

class MemoryImageSource(AbstractImageSource):
    def __init__(self, image: Image.Image, description: str):
        self._image = image
        self.description = description

    def get_image(self) -> Image.Image:
        return self._image

    def get_description(self) -> str:
        return self.description

# ==================
# IMAGE EXPLORER
# ==================
class ImageExplorer:
    def __init__(self, config: Dict, image_dir: Path = None):
        self.config = config
        self.image_sources: List[AbstractImageSource] = []
        self._clipboard_source: Optional[MemoryImageSource] = None
        self.current_index = 0
        self.is_kitty = os.environ.get("TERM") == "xterm-kitty"
        self.ocr_cancelled = False
        self._init_sources(image_dir)

    def _init_sources(self, image_dir: Optional[Path]):
        """Initialize image sources from directory and clipboard"""
        # Load directory images
        default_dir = Path(self.config["image_settings"]["default_directory"]).expanduser()
        default_dir.mkdir(parents=True, exist_ok=True)

        self.image_dir = image_dir or default_dir
        pattern = self.config["image_settings"]["filename_pattern"]

        file_paths = sorted(
            self.image_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        file_sources = [FileImageSource(p) for p in file_paths]

        # Check clipboard
        clipboard_img = self._get_clipboard_image()
        if clipboard_img:
            self._clipboard_source = MemoryImageSource(clipboard_img, "clipboard")
            # Only add if different from first file source
            if file_sources and self._images_identical(self._clipboard_source, file_sources[0]):
                self.image_sources = file_sources
            else:
                self.image_sources = [self._clipboard_source] + file_sources
        else:
            self.image_sources = file_sources

    def _get_clipboard_image(self) -> Optional[Image.Image]:
        """Get image from clipboard using PIL"""
        try:
            return ImageGrab.grabclipboard()
        except Exception as e:
            print(f" Clipboard error: {str(e)}")
            return None

    def _images_identical(self, a: AbstractImageSource, b: AbstractImageSource) -> bool:
        """Compare two image sources for pixel equality"""
        img_a = a.get_image()
        img_b = b.get_image()

        # Quick size check
        if img_a.size != img_b.size:
            return False

        # Hash comparison
        return self._image_hash(img_a) == self._image_hash(img_b)

    def _image_hash(self, img: Image.Image) -> str:
        """Generate hash for image comparison"""
        return hashlib.sha256(img.tobytes()).hexdigest()


    def current_image(self) -> AbstractImageSource:
        return self.image_sources[self.current_index]

    def show_image(self):
        """Display current image preview"""
        source = self.current_image()
        if self.is_kitty:
            print("\x1b[2J\x1b[H", end="", flush=True)
            self._kitty_show_image(source.get_image())
        print(f"\n Current image: {source.get_description()}")

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_sources)
        self.show_image()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_sources)
        self.show_image()

    # Modified kitty_show_image to accept Image directly
    def _kitty_show_image(self, img: Image.Image):
        """Display image preview using Kitty terminal's icat"""
        try:
            img = self.get_preview_image(img)
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            subprocess.run(
                ["kitty", "+kitten", "icat", "--align=left", "--stdin=yes"],
                input=img_buffer.getvalue(),
                check=True
            )
        except Exception as e:
            print(f"\n Image display error: {str(e)}")

    def get_preview_image(self, img: Image.Image) -> Image.Image:
        """Get resized image for preview"""
        settings = self.config["preview_settings"]
        return resize_image(
            img,
            settings.get("max_width"),
            settings.get("max_height")
        )

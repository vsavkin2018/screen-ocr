from PIL import Image
import io
from abc import ABC, abstractmethod
from typing import List, AsyncGenerator, Tuple, Optional, Dict

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


# =======================
# ABSTRACT CLASS FOR OCR
# =======================
class BaseEngine(ABC):
    @abstractmethod
    async def stream_ocr(self, image: Image.Image) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def cancel(self):
        pass

    def get_max_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (max_width, max_height) for this engine. None means no limit."""
        return (None, None)  # Default: no resizing

    def prepare_image(self, img: Image.Image) -> Image.Image:
        """Resize image according to engine requirements"""
        max_width, max_height = self.get_max_dimensions()
        return resize_image(img, max_width, max_height)

    async def stream_chat(self, message: str, context: dict) -> AsyncGenerator[str, None]:
        """Default chat implementation"""
        yield "\nChat not supported\n"

class DummyEngine(BaseEngine):
    def __init__(self):
        pass
    async def stream_ocr(self, image: Image.Image) -> AsyncGenerator[str, None]:
        yield("Nothing to see here!")
    async def cancel(self):
        pass

from PIL import Image
import asyncio
import io
from typing import AsyncGenerator, Tuple, Optional
import pytesseract
from ocr_base import BaseEngine, resize_image

class TesseractEngine(BaseEngine):
    def __init__(self, config: dict):
        self.config = config.get("tesseract", {})
        self._cancelled = False

    def get_max_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        return (
            self.config.get("max_width", None),
            self.config.get("max_height", None)
        )

    async def cancel(self):
        """Set cancellation flag to stop processing"""
        self._cancelled = True

    async def stream_ocr(self, image: Image.Image) -> AsyncGenerator[str, None]:
        """Tesseract OCR implementation with async wrapper"""
        self._cancelled = False
        processed_image = self.prepare_image(image)
        
        try:
            # Run blocking Tesseract call in thread
            result = await asyncio.to_thread(
                self._run_tesseract,
                processed_image
            )
            
            if self._cancelled:
                yield "\n OCR aborted"
                return
                
            yield result
            
        except pytesseract.TesseractNotFoundError:
            yield "\n OCR Error: Tesseract not found. Install it and add to PATH"
        except Exception as e:
            yield f"\n OCR Error: {str(e)}"

    def _run_tesseract(self, image: Image.Image) -> str:
        """Synchronous Tesseract processing"""
        return pytesseract.image_to_string(
            image,
            lang=self.config.get("lang", "eng"),
            config=self._get_tesseract_config()
        )

    def _get_tesseract_config(self) -> str:
        """Build Tesseract config string from settings"""
        psm = self.config.get("psm", 3)
        oem = self.config.get("oem", 1)
        return f'--psm {psm} --oem {oem}'

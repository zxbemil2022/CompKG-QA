from src.plugins._ocr import OCRPlugin

_OCR_INSTANCE: OCRPlugin | None = None


def _get_ocr() -> OCRPlugin:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        _OCR_INSTANCE = OCRPlugin()
    return _OCR_INSTANCE


class _LazyOCRProxy:
    """延迟初始化 OCR，避免服务启动时阻塞/异常中断。"""

    def __getattr__(self, item):
        return getattr(_get_ocr(), item)


ocr = _LazyOCRProxy()

__all__ = ["ocr"]

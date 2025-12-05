"""
SGAPS-MAE Client Package
Lightweight client-side components for pixel sampling
"""

from .minimal_client import MinimalGameClient, ClientConfig
from .pixel_extractor import PixelExtractor
from .compressor import PacketCompressor, PacketDecompressor

__all__ = [
    'MinimalGameClient',
    'ClientConfig',
    'PixelExtractor',
    'PacketCompressor',
    'PacketDecompressor',
]

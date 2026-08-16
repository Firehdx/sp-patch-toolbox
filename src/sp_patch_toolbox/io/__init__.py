"""Format-aware region readers for spatial-proteomics images."""

from .readers import IMAGE_SUFFIXES, BaseImageReader, open_image_reader

__all__ = ["IMAGE_SUFFIXES", "BaseImageReader", "open_image_reader"]

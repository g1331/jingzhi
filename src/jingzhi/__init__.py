"""Local-first multimodal desktop context companion."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jingzhi")
except PackageNotFoundError:
    __version__ = "0+unknown"

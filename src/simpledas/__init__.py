from .simpleDASreader import *
from . import h5pydict
import importlib.metadata

__all__ = ['simpleDASreader', 'h5pydict']
__version__ = importlib.metadata.version(__name__)
__author__ = importlib.metadata.metadata('simpledas')['Author']

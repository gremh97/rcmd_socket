"""
RCMD (Remote Command)

A command-line interface tool for managing and running models on remote boards.
"""

__version__ = "0.1.0"

from .cli import RCMD, cli
from .client import Client

__all__ = ['RCMD', 'cli', 'Client']
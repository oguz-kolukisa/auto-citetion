"""Output generation: markdown reports and paper files."""

from .report import generate_report
from .paper_files import generate_paper_files

__all__ = ["generate_report", "generate_paper_files"]

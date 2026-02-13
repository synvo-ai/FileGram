"""Allow running as ``python -m filegramplugin``."""

import sys

from .main import main

cfg = sys.argv[1] if len(sys.argv) > 1 else None
main(cfg)

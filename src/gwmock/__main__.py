"""Main entry point for gwmock package."""

from __future__ import annotations

if __name__ == "__main__":
    from gwmock.utils.log import setup_logger

    setup_logger(print_version=True)

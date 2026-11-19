"""Entry point for uvicorn: `uvicorn gateway.main:app`."""

import logging

from gateway.app import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()

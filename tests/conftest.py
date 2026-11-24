import logging

import pytest


@pytest.fixture(autouse=True)
def close_run_log():
    """run() attaches a file handler to the shared logger; release it after each test."""
    yield
    log = logging.getLogger("papertrail")
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)

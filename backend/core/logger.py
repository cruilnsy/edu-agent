# backend/core/logger.py
# Thin wrapper over the standard logging module that supports structured
# "event name + key=value" log lines.

import logging
import sys

from backend.config import get_settings


class _Logger:
    """Structured logger: logger.info("event.name", key=value) instead of hand-built strings.

    The leading underscore marks it as internal; modules get an instance through get_logger().
    """

    def __init__(self, name: str):
        # name is usually __name__, so each line shows which module wrote it
        self._log = logging.getLogger(name)

    def _fmt(self, event: str, **kw) -> str:
        """Format an event and its fields into one line.

        Example: _fmt("user.login", user_id="u1") -> "user.login | user_id='u1'"
        Values use repr(), so strings are quoted. Never pass secrets as fields.
        """
        if kw:
            return event + " | " + " ".join(f"{k}={v!r}" for k, v in kw.items())
        return event

    def debug(self, event: str, **kw):
        self._log.debug(self._fmt(event, **kw))

    def info(self, event: str, *args, **kw):
        """Accepts both styles: info("event", key=value) and the classic info("template %s", value)."""
        if args:
            self._log.info(event, *args)
        else:
            self._log.info(self._fmt(event, **kw))

    def warning(self, event: str, *args, **kw):
        if args:
            self._log.warning(event, *args)
        else:
            self._log.warning(self._fmt(event, **kw))

    def error(self, event: str, **kw):
        """Pass exc_info=True to include the current traceback."""
        exc_info = kw.pop("exc_info", False)
        self._log.error(self._fmt(event, **kw), exc_info=exc_info)

    def critical(self, event: str, **kw):
        self._log.critical(self._fmt(event, **kw))


def configure_logging() -> None:
    """Configure root logging once at startup (called from main.py)."""
    settings = get_settings()
    # "INFO" -> logging.INFO; an unknown level name falls back to INFO
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        level=level,
        force=True,                  # Replace any handlers configured earlier
    )
    # Silence noisy third-party INFO logs
    for noisy in ("sqlalchemy.engine", "sqlalchemy.pool", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> _Logger:
    """Each module calls get_logger(__name__) to get its own logger."""
    return _Logger(name)

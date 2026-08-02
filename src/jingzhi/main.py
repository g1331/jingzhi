from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from jingzhi.config import Settings
from jingzhi.ui import run_app


def main() -> None:
    settings = Settings.from_env()
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    raise SystemExit(run_app(settings))


if __name__ == "__main__":
    main()

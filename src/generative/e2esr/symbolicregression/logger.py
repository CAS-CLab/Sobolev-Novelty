# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# licenses/e2esr.txt file in the root directory of this repository.
# Modified for repository integration: license location and portable imports.


import logging
import time
import os
import re
from pathlib import Path
from datetime import timedelta
from logging.handlers import RotatingFileHandler


# class LogFormatter:
#     def __init__(self):
#         self.start_time = time.time()

#     def format(self, record):
#         elapsed_seconds = round(record.created - self.start_time)

#         prefix = "%s - %s - %s" % (
#             record.levelname,
#             time.strftime("%x %X"),
#             timedelta(seconds=elapsed_seconds),
#         )
#         message = record.getMessage()
#         message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
#         return "%s - %s" % (prefix, message) if message else ""


class LogFormatter(logging.Formatter):
    color_dict = {
        "DEBUG": "\033[0;37m{}\033[0m",
        "INFO": "\033[0;34m{}\033[0m",
        "NOTE": "\033[1;38;5;46m{}\033[0m",
        "WARNING": "\033[1;48;5;220m{}\033[0m",
        "ERROR": "\033[0;30;41m{}\033[0m",
        "CRITICAL": "\033[0;30;45m{}\033[0m",
    }

    def __init__(
        self, exp_name, colorful=False, start_time=None, time_format="%y-%m-%d %H:%M:%S"
    ):
        super().__init__()
        self.exp_name = exp_name
        self.colorful = colorful
        self.start_time = start_time or time.time()
        self.time_format = time_format

    def format(self, record):
        prefixes = [
            self.exp_name,
            record.name.split(".")[-1],
            record.levelname[0],  # D, I, N, W, E, C
            time.strftime(self.time_format),
            str(timedelta(seconds=record.created - self.start_time)),
        ]
        prefix = f"[{'|'.join([str(p) for p in prefixes if str(p).strip()])}]"
        if record.levelname in ["WARNING", "ERROR", "CRITICAL"]:
            path = os.path.relpath(record.pathname, os.getcwd())
            prefix += f" ({path}:{record.lineno})"
        message = record.getMessage() or ""
        message = message.replace("\n", "\n" + " " * len(prefix + " "))
        # message = message.replace("\n", "\n" + " " * 8)
        if self.colorful:
            return (
                self.color_dict.get(record.levelname, "{}").format(prefix)
                + " "
                + message
            )
        else:
            return prefix + " " + re.sub(r"\033\[[\d;]+m", "", message)


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    start_time = time.time()
    name = 'E2ESR'

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = "%s-%i" % (filepath, rank)
        name = Path(filepath).parent.name
        log_formatter = LogFormatter(
            name,
            colorful=False,
            start_time=start_time,
            time_format="%b%d %H:%M:%S",
        )
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filepath,
            mode="a",
            maxBytes=int(50.0 * 1024 * 1024),
            backupCount=100,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    log_formatter = LogFormatter(name, colorful=True, start_time=start_time, time_format="%b%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        for handler in logger.handlers:
            if hasattr(handler.formatter, "start_time"):
                handler.formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger

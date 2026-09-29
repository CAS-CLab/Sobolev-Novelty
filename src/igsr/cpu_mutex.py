"""
Process-wide mutex for heavyweight, CPU-bound sections.

Usage:
```python
with cpu_mutex():  # Blocks until every other holder releases.
    heavy_step()
```
"""

import contextlib
import fcntl
import os
import time
from typing import Iterator, Optional

_DEFAULT_LOCKFILE = "/tmp/heavy_step.lock"


@contextlib.contextmanager
def cpu_mutex(
    lockfile: str = _DEFAULT_LOCKFILE,
    *,
    wait_msg_interval: Optional[float] = None,
    identifier: Optional[str] = None,
) -> Iterator[None]:
    """
    Acquire an *exclusive* advisory lock before entering the context.

    Args:
        lockfile (str, optional): Path to the lock file. Any shared, local FS path is fine.
            The file is created automatically and may remain empty forever.
        wait_msg_interval (float | None):
            Seconds between "still waiting..." prints while the lock is busy.
            Also prints "got lock, starting." when finally acquired.
            None --> silent (default).
        identifier (str | None):
            Extra tag printed with the waiting message.

    Behaviour:
    * Blocks until this process owns the lock.
    * The kernel releases the lock automatically if the process crashes or `exec`s another program without
        `FD_CLOEXEC` cleared.
    * The lock is released automatically:
        - on leaving the context (normal or exceptional exit)
        - if the process receives SIGTERM / SIGKILL / crashes
    """
    tag = f" [{identifier}]" if identifier else ""

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    fd = os.open(lockfile, flags, 0o666)
    try:
        start = last_msg = time.monotonic()
        while True:
            try:
                fcntl.flock(
                    fd,
                    fcntl.LOCK_EX | (fcntl.LOCK_NB if wait_msg_interval else 0),
                )
                if wait_msg_interval is not None:
                    print(f"cpu_mutex{tag}: got lock, starting.")
                break  # Got it.
            except BlockingIOError:
                if wait_msg_interval is not None:
                    now = time.monotonic()
                    if now - last_msg >= wait_msg_interval:
                        waited = int(now - start)
                        print(f"cpu_mutex{tag}: waiting for lock ({waited}s)…")
                        last_msg = now
                time.sleep(min(0.1, wait_msg_interval or 0.1))
            except InterruptedError:
                continue  # Retry on signal.

        yield  # Critical.
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

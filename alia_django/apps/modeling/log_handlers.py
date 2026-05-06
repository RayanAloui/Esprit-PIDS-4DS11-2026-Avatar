import logging
import os

class RealTimeFileHandler(logging.FileHandler):
    """
    A custom FileHandler that forces the operating system to flush the buffer
    to the disk immediately upon every log emit. This solves the issue on Windows
    where the file modification time and contents aren't updated in real-time
    until the file handle is closed.
    """
    def emit(self, record):
        super().emit(record)
        if self.stream:
            self.stream.flush()
            # Force write to physical disk
            try:
                os.fsync(self.stream.fileno())
            except OSError:
                pass

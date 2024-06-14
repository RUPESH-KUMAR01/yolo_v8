import logging
import os
import sys
from types import SimpleNamespace
RANK = int(os.getenv("RANK", -1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format
from psutil import WINDOWS
from tqdm import tqdm as tqdm_original
LOGGING_NAME="custom_modified"
version="1.0.0"
TESTS_RUNNING=False

def set_logging(name="LOGGING_NAME", verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support, ensuring compatibility across different
    environments.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    # Configure the console (stdout) encoding to UTF-8, with checks for compatibility
    formatter = logging.Formatter("%(message)s")  # Default formatter
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                return (super().format(record))

        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            LOGGER.info(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)
class SimpleClass:
    """Ultralytics SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
    


    
class IterableSimpleNamespace(SimpleNamespace):
    """Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)
    

class TQDM(tqdm_original):
    """
    Custom Ultralytics tqdm class with different default arguments.

    Args:
        *args (list): Positional arguments passed to original tqdm.
        **kwargs (any): Keyword arguments, with custom defaults applied.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize custom Ultralytics tqdm class with different default arguments.

        Note these can still be overridden when calling TQDM.
        """
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)  # logical 'and' with default value if passed
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)

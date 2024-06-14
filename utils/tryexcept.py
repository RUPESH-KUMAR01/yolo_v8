import contextlib


class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>>    # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>>     # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print((f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True

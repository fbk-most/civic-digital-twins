"""The atomic module provides thread-safe atomic operations for integer values.

It implements an atomic integer counter class similar to Go's atomic.Int64.
"""

import threading


class Int:
    """
    A thread-safe integer class supporting atomic operations.

    This class provides atomic operations on integer values by using
    a lock to ensure thread-safety. It allows incrementing, adding,
    and reading values without race conditions in multithreaded environments.
    """

    def __init__(self):
        """Initialize an atomic integer with a value of 0."""
        self.__value = 0
        self.__lock = threading.Lock()

    def add(self, value: int) -> int:
        """
        Atomically add a value to the current value.

        Args:
            value (int): The value to add.

        Returns
        -------
            int: The new value after addition.
        """
        with self.__lock:
            self.__value += value
            return self.__value

    def load(self) -> int:
        """
        Atomically load and return the current value.

        Returns
        -------
            int: The current value.
        """
        with self.__lock:
            return self.__value

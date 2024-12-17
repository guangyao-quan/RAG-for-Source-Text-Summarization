"""
This module provides a decorator to track the execution time of functions.

Functions:
----------

1. `time_tracker(func)`:
    - A decorator function that wraps another function to track and print its execution time.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function that tracks and prints execution time.

Usage Example:
--------------

    @time_tracker
    def example_function():
        # Function implementation
        pass

    # The function `example_function` will now print its execution time after running.
"""

from functools import wraps
import time


def time_tracker(func):
    """
    Decorator to track the execution time of a function.

    This decorator prints the name of the function being executed and the time it took to complete.

    Args:
        func (function): The function whose execution time is to be tracked.

    Returns:
        function: The wrapped function that tracks and prints execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper function to calculate and print the execution time of the decorated function.

        Args:
            *args: Positional arguments passed to the decorated function.
            **kwargs: Keyword arguments passed to the decorated function.

        Returns:
            tuple: A tuple containing the result of the function execution and the elapsed time.
        """
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the decorated function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")  # Print the execution time
        return result, elapsed_time  # Return the result and the elapsed time

    return wrapper

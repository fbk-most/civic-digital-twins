"""Tests for the dt_model.engine.atomic.Int type."""

# SPDX-License-Identifier: Apache-2.0

import threading

from dt_model.engine.atomic import Int


def test_atomic_int_basic():
    """Test basic operations of atomic Int."""
    counter = Int()
    assert counter.load() == 0  # Initial value

    # Test add operation
    assert counter.add(1) == 1  # Returns new value
    assert counter.load() == 1  # Verify current value

    # Test larger increments
    assert counter.add(10) == 11
    assert counter.load() == 11


def test_atomic_int_thread_safety():
    """Test thread safety of atomic Int."""
    counter = Int()
    iterations = 1000
    threads = 10

    def increment_counter():
        for _ in range(iterations):
            counter.add(1)

    # Create and start threads
    thread_list = []
    for _ in range(threads):
        thread = threading.Thread(target=increment_counter)
        thread_list.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in thread_list:
        thread.join()

    # Check final counter value
    # If counter is thread-safe, value should be threads * iterations
    assert counter.load() == threads * iterations

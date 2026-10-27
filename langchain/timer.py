
import time
import threading
from functools import wraps


def show_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        stop_event = threading.Event()
        start_time = time.perf_counter()

        def timer():
            while not stop_event.wait(1):
                elapsed = time.perf_counter() - start_time
                print(
                    f"\rElapsed: {elapsed:.0f}s",
                    end="",
                    flush=True
                )

        thread = threading.Thread(target=timer, daemon=True)
        thread.start()

        try:
            return func(*args, **kwargs)
        finally:
            stop_event.set()
            thread.join()

            elapsed = time.perf_counter() - start_time
            print(f"\rCompleted in {elapsed:.2f} seconds.   ")

    return wrapper

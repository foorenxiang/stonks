import math
import time


def exec_time(func):
    def exec_time_wrapper(*args, **kwargs):
        start_time = time.monotonic()
        func(*args, **kwargs)
        end_time = time.monotonic()
        time_taken = end_time - start_time
        mins, secs = math.floor(time_taken / 60), math.floor(time_taken % 60)
        print(f"Time taken: {mins}mins {secs}secs")

    return exec_time_wrapper

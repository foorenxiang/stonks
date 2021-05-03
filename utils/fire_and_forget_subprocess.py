# referenced from https://stackoverflow.com/questions/17937249/fire-and-forget-a-process-from-a-python-script

import asyncio
import time


def fire_and_forget(func):
    def wrapper_function(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, func, *args, *kwargs)

    return wrapper_function


@fire_and_forget
def foo():
    time.sleep(1)
    print("foo() completed")


if __name__ == "__main__":
    print("Hello")
    foo()
    print("I didn't wait for foo()")

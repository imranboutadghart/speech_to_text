import time

class Timer:
    def __init__(self):
        self.start = None
        self.indent = 0
        self.end = None

    def __enter__(self):
        return self

    def __str__(self):
        return str(self.interval)

    def __float__(self):
        return self.interval

    def __int__(self):
        return int(self.interval)

    def __repr__(self):
        return f"{self.interval:.2f} seconds"

    def start(self):
        self.start = time.perf_counter()
    
    def stop(self):
        self.end = time.perf_counter()
    def print(self):
        if (self.start is None) or (self.end is None) or (self.end < self.start):
            raise ValueError("Timer not started or stopped")
        print(f"{' ' * self.indent}Execution time: {self.interval:.2f} seconds")
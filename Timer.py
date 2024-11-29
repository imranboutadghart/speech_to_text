import time

class Timer:
    def __init__(self):
        self.start_time = []
        self.end_time = []
        self.indent = 0

    def start(self):
        self.start_time.append(time.time())
        self.indent += 1

    def end(self, message):
        self.end_time.append(time.time())
        if (len(self.start_time) < len(self.end_time)):
            print("Error: Timer not started")
            return
        print("  " * self.indent, end="")
        self.indent -= 1
        print("\033[93m"f"{message}: {self.end_time.pop() - self.start_time.pop()} seconds" + "\033[0m")
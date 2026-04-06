import pandas as pd

class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.buffer = []

    def add(self, row):
        self.buffer.append(row)

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        df.to_csv(self.path, mode='a', index=False, header=False)
        self.buffer.clear()
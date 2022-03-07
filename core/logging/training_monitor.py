import json
import matplotlib.pyplot as plt
import numpy as np
import os

class TrainingMonitor:

    def __init__(self, jpg_path, json_path, start=0):
        self.jpg_path = jpg_path
        self.json_path = json_path
        self.start = start

    def init(self):
        self.H = {}
        if os.path.exists(self.json_path):
            self.H = json.loads(open(self.json_path).read())
            if self.start > 0:
                for key in self.H.keys():
                    self.H[key] = self.H[key][:self.start]

    def update(self, logs={}):
        for key, value in logs.items():
            data = self.H.get(key, [])
            data.append(float(value))
            self.H[key] = data

        file = open(self.json_path, "w")
        file.write(json.dumps(self.H))
        file.close()

        N = np.arange(0, len(self.H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        for key, values in self.H.items():
            plt.plot(N, values, label=key)
        plt.title(f"Training Monitor [Epoch {len(self.H['loss'])}]")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(self.jpg_path)
        plt.close()

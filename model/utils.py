import torch
import shutil

class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

def save_checkpoint(state, is_best, checkpoint):
    pass

def load_checkpoint(checkpoint, model, optimizer=None):
    pass

import torch
import shutil
import json
import os

class Params():

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent = 4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

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
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exist(checkpoint):
        print("Checkpoint Directory does not exist. Making Directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists.")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def print_stat_t(tensor_name, tensor):
    device = "GPU" if tensor.is_cuda else "CPU"
    print("Tensor [" + tensor_name + "] on " + device + ". size:", tensor.size(), "dtype:", tensor.dtype)
    print("Tensor [" + tensor_name + "] stat: max {:.3f}, min {:.3f}, mean {:.3f}, median {:.3f}, std {:.3f}".format(tensor.max().item(), tensor.min().item(), tensor.mean().item(), tensor.median().item(), tensor.std().item()))

def print_model_params(model):
    print("#total params:", sum(p.numel() for p in model.parameters()), end='')
    print(" | #trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

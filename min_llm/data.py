import numpy as np
import torch
from torch.utils.data import Dataset
SEED = 42
rng = np.random.default_rng(SEED)

def get_batch(x: np.array,
              batch_size: int,
              context_length: int,
              device: str = "cuda:0"):
    
    if not torch.cuda.is_available() and "cuda" in device:
        raise RuntimeError("Cuda not found. Pleae set device = cpu")
    device = torch.device(device)

    while True:
        start_indices = rng.choice(len(x) - context_length, size = (batch_size,), replace = False)
        offsets = np.arange(context_length)
        indices = start_indices[:,np.newaxis] + offsets 
        yield torch.from_numpy(x[indices]).long().to(device), torch.from_numpy(x[indices + 1]).long().to(device)

    
class SequenceDataset(Dataset):

    def __init__(self, file_path, context_length, stride = None):

        self.data = np.load(file_path, mmap_mode='r')
        self.context_length = context_length
        self.stride = stride if stride is not None else 1

        self.max_start = len(self.data) - context_length
        self.num_samples = max(0, (self.max_start - 1) // self.stride + 1) 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        sequence = self.data[start_idx : start_idx + self.context_length + 1].copy()
        sequence = torch.from_numpy(sequence).long()
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        return input_ids, target_ids





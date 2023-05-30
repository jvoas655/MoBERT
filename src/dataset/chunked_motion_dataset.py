import atexit
import math
from multiprocessing import Pool, Value
from pathlib import Path
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from threading import Thread, Event
import queue

class ChunkedMotionDataset(Dataset):
    def _load_and_chunk_motions(self, args):
        all_files, chunk_size, overlap = args
        all_chunked_data = []
        all_chunked_masks = []
        for file in all_files:
            data = np.load(file)
            data_len = len(data)
            
            for i in range(0, data_len, chunk_size - overlap):
                chunked_data = data[i: i+chunk_size]
                chunked_masks = [1] * len(chunked_data) + [0] * (chunk_size - len(chunked_data))
                if (len(chunked_data) < chunk_size):
                    chunked_data = np.pad(chunked_data, ((0, chunk_size - len(chunked_data)), (0, 0)), 'constant', constant_values=(0, 0))
                all_chunked_data.append(np.expand_dims(chunked_data, axis = 0))
                all_chunked_masks.append(np.expand_dims(np.array(chunked_masks), axis = 0))
        return np.concatenate(all_chunked_data, axis=0), np.concatenate(all_chunked_masks, axis=0)
    def __init__(self, exp_name, split, datapath, limit_size, batch_size, chunk_size, overlap, n_workers = 1, transform=None):
        assert chunk_size - overlap > 0
        self.base_path = Path("../MotionDataset/cache")
        os.makedirs(self.base_path, exist_ok=True)
        self.all_data = []
        self.all_masks = []
        all_files = []
        self.exp_name = exp_name
        self.split = split
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_borders = [0]
        print("Gathering Chunked Motion Dataset Files")
        filei = 0
        for root, dir, files in os.walk(datapath):
            for file in files:
                if (Path(file).suffix != ".npy"):
                    continue
                all_files.append(Path(root) / file)
                filei += 1
                if (filei >= limit_size and limit_size != -1):
                    break
        self.chunk_count = 0
        for batch_i in range(0, len(all_files), batch_size):
            batch_data = []
            batch_masks = []
            if (os.path.exists(self.base_path / f"{exp_name}_{split}_{chunk_size}_{overlap}_batch_{self.chunk_count}_all_masks.npy")):
                self.chunk_count += len(np.load(self.base_path / f"{exp_name}_{split}_{chunk_size}_{overlap}_batch_{self.chunk_count}_all_masks.npy"))
                self.chunk_borders.append(self.chunk_count)
                continue
            proc_files = {}
            for i in range(n_workers):
                proc_files[i] = []
            for filei in range(batch_i, min(batch_i + batch_size, len(all_files))):
                proc_files[filei % n_workers].append(all_files[filei])
            args = []
            for i in proc_files:
                args.append((proc_files[i], chunk_size, overlap))
            with Pool(n_workers) as pool:
                for chunked_data, chunked_masks in pool.imap_unordered(self._load_and_chunk_motions, args):
                    batch_data.append(chunked_data)
                    batch_masks.append(chunked_masks)
            batch_data = np.concatenate(batch_data, axis=0)
            batch_masks = np.concatenate(batch_masks, axis=0)
            np.save(self.base_path / f"{exp_name}_{split}_{chunk_size}_{overlap}_batch_{self.chunk_count}_all_data.npy", batch_data)
            np.save(self.base_path / f"{exp_name}_{split}_{chunk_size}_{overlap}_batch_{self.chunk_count}_all_masks.npy", batch_masks)
            self.chunk_count += len(batch_masks)
            self.chunk_borders.append(self.chunk_count)
        self._get_and_load_chunk_data(0)
    def __len__(self):
        return self.chunk_count

    def _get_and_load_chunk_data(self, idx):
        for i, border in enumerate(self.chunk_borders):
            if (self.chunk_borders[i + 1] > idx):
                break
        self.offset = border
        self.threshold = self.chunk_borders[i + 1]
        
        self.data = np.load(self.base_path / f"{self.exp_name}_{self.split}_{self.chunk_size}_{self.overlap}_batch_{border}_all_data.npy")
        self.masks = np.load(self.base_path / f"{self.exp_name}_{self.split}_{self.chunk_size}_{self.overlap}_batch_{border}_all_masks.npy")

    def __getitem__(self, idx):
        if not (idx > self.offset and idx < self.threshold):
            self._get_and_load_chunk_data(idx)
        return {
            "motion_chunk": self.data[idx - self.offset],
            "motion_mask": self.masks[idx - self.offset]
        }

if __name__ == "__main__":
    
    dtrain = ChunkedMotionDataset("std", "train", Path("../MotionDataset/T2MEval_dataset/train_data/"), -1, 20000, 16, 4, 32)
    #dtest = ChunkedMotionDataset("std", "test", Path("../MotionDataset/T2MEval_dataset/test_data/"), -1, 20000, 16, 4, 32)
    #dval = ChunkedMotionDataset("std", "val", Path("../MotionDataset/T2MEval_dataset/val_data/"), -1, 20000, 16, 4, 32)
    train_data = DataLoader(dtrain, batch_size=256, shuffle=False)
    for batch_i, batch_data in enumerate(train_data):
        batch_motion_chunks = batch_data["motion_chunk"]
        batch_motion_mask = batch_data["motion_mask"]
        print(batch_i, batch_motion_chunks.shape)
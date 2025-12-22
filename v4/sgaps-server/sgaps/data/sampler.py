import torch
from torch.utils.data import Sampler
import numpy as np
import math

class SortedBatchSampler(Sampler):
    """
    A Sampler that:
    1. Sorts the dataset by pixel count (or sequence length).
    2. Groups sorted samples into batches (minimizing padding within each batch).
    3. Shuffles the ORDER of the batches (maintaining IID assumption across the epoch).
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        """
        Args:
            dataset: SGAPSDataset instance (must have .pixel_counts logic or similar logic to get lengths).
                     If not SGAPSDataset, assumes dataset has a compatible API or we might need a workaround.
            batch_size: Size of batch.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # We need pixel counts. SGAPSDataset populated this.
        # If 'dataset' is a Subset (from random_split), we need to access the underlying dataset and indices
        if isinstance(dataset, torch.utils.data.Subset):
            # Map subset indices to original dataset indices
            indices = dataset.indices
            # Access underlying dataset's pixel counts
            # Note: This assumes underlying dataset is SGAPSDataset
            full_pixel_counts = np.array(dataset.dataset.pixel_counts)
            self.pixel_counts = full_pixel_counts[indices]
            self.original_indices = np.array(indices)
        else:
            # Assume it's the full SGAPSDataset
            if not hasattr(dataset, 'pixel_counts'):
                # Fallback: Can't sort efficiently if counts aren't cached.
                # Just use random order? Or error out?
                # Raising warning and falling back to random indices logic (though ineffective for this class purpose)
                # But better to raise error as this class is specific.
                raise ValueError("Dataset must have 'pixel_counts' attribute for SortedBatchSampler.")
            
            self.pixel_counts = np.array(dataset.pixel_counts)
            self.original_indices = np.arange(len(dataset))
            
        # 1. Sort indices by pixel count
        # Argsort gives indices relative to the 'pixel_counts' array we just extracted
        # These are exactly the indices into 'dataset' (whether subset or full) that DataLoader expects
        self.sorted_indices = np.argsort(self.pixel_counts)

    def __iter__(self):
        # 2. Chunk into batches
        # We re-calculate batches each iter to allow for Shuffle of Batches logic to be clean,
        # but the grouping (content of batches) is deterministic based on length.
        
        batches = []
        n = len(self.sorted_indices)
        
        for i in range(0, n, self.batch_size):
            batch = self.sorted_indices[i : i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            batches.append(batch)
            
        # 3. Shuffle the order of batches
        # This is critical for training stability
        np.random.shuffle(batches)
        
        # Yield all batches
        for batch in batches:
            yield list(batch)

    def __len__(self):
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        else:
            return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size

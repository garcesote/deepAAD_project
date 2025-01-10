from torch.utils.data.sampler import Sampler
import numpy as np

# Randomize batch order for present batches from different trials to the network
class BatchRandomSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_batches = len(data_source) // batch_size

    def __iter__(self):
        batch_indices = np.arange(self.num_batches)
        np.random.shuffle(batch_indices)
        for batch_idx in batch_indices:
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            # return the desired idx to let the dataloader access samples efficiently with yield
            yield np.arange(start, end)

    def __len__(self):
        return self.num_batches
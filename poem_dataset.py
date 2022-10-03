import torch
from lib import data_utils
from torch.utils.data.dataset import Dataset


class PoemDataset(Dataset):
    def __init__(self, inp_file, ctr_file, out_file) -> None:
        self.inp = data_utils.unpickle_file(inp_file)
        self.ctr = data_utils.unpickle_file(ctr_file)
        self.out = data_utils.unpickle_file(out_file)
    
    def __len__(self):
        return len(self.ctr)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        inp = torch.FloatTensor(self.inp[index])
        ctr = torch.FloatTensor(self.ctr[index])
        out = torch.FloatTensor(self.out[index])

        sample = (inp, ctr, out)

        return sample

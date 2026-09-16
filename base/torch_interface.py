import torch
import numpy as np

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()

        indices = np.vstack((coo.row, coo.col))
        i = torch.tensor(indices, dtype=torch.int64)

        #i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
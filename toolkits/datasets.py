from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import h5py


def train_test_split(speedCollection, volCollection, train_size=None, test_size=None, random_number=42):
    np.random.seed(random_number)
    if train_size:
        trainDataIdx = np.random.choice(
            len(speedCollection),
            int(train_size * len(speedCollection)),
            replace=False
        )
        testDataIdx = set([i for i in range(len(speedCollection))]) -\
                      set(trainDataIdx)
    
    elif test_size:
        testDataIdx = np.random.choice(
            len(speedCollection),
            int(test_size * len(speedCollection)),
            replace=False
        )
        trainDataIdx = set([i for i in range(len(speedCollection))]) -\
                       set(testDataIdx)
        
    trainSpeed = list(pd.Series(speedCollection)[list(trainDataIdx)])
    trainVol = list(pd.Series(volCollection)[list(trainDataIdx)])
    testSpeed = list(pd.Series(speedCollection)[list(testDataIdx)])
    testVol = list(pd.Series(volCollection)[list(testDataIdx)])

    return trainSpeed, trainVol, testSpeed, testVol


################################## Datasets inherited from torch.utils.data.Dataset ##################################
class CNNDataset(Dataset):
    def __init__(
            self,
            speed_data: list = None,
            volume_data: list = None,
            load_ckpt: bool = None,
            mode: str = None,
            ckpt_dir: str = 'C:/Users/Home/PythonProjects/hwyTrafficPred/toolkits/cnndataset'
    ) -> None:
        if (speed_data):
            self.speedFeature = [speed_data[x][0] for x in range(len(speed_data))]
            self.volFeature = [volume_data[x][0] for x in range(len(volume_data))]
            self.speedLabels = [speed_data[x][1][[1],:] for x in range(len(speed_data))]
            self.volLabels = [volume_data[x][1][[1],:] for x in range(len(volume_data))]
        
        else:
            if (load_ckpt) and (mode == 'train'):
                with h5py.File(f"{ckpt_dir}/{mode}/{mode}_speed_feature.h5", 'r') as file:
                    self.speedFeature = file[f"{mode}_speed_feature"][:]
                with h5py.File(f"{ckpt_dir}/{mode}/{mode}_volume_feature.h5", 'r') as file:
                    self.volFeature = file[f"{mode}_volume_feature"][:]
                with h5py.File(f"{ckpt_dir}/{mode}/{mode}_speed_label.h5", 'r') as file:
                    self.speedLabels = file[f"{mode}_speed_label"][:]
                with h5py.File(f"{ckpt_dir}/{mode}/{mode}_volume_label.h5", 'r') as file:
                    self.volLabels = file[f"{mode}_volume_label"][:]
            
            elif (load_ckpt) and (mode == 'test'):
                with h5py.File(f"{ckpt_dir}/{mode}/{mode}_speed_feature.h5", 'r') as file:
                    self.speedFeature = file[f"{mode}_speed_feature"][:]
                with h5py.File(f"{ckpt_dir}/{mode}/{mode}_volume_feature.h5", 'r') as file:
                    self.volFeature = file[f"{mode}_volume_feature"][:]
                with h5py.File(f"{ckpt_dir}/{mode}/{mode}_speed_label.h5", 'r') as file:
                    self.speedLabels = file[f"{mode}_speed_label"][:]
                with h5py.File(f"{ckpt_dir}/{mode}/{mode}_volume_label.h5", 'r') as file:
                    self.volLabels = file[f"{mode}_volume_label"][:]

    def __len__(self) -> int:
        return len(self.speedFeature)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        f1 = torch.tensor(self.speedFeature[idx], dtype=torch.float).unsqueeze(0)
        f2 = torch.tensor(self.volFeature[idx], dtype=torch.float).unsqueeze(0)
        l1 = torch.tensor(self.speedLabels[idx], dtype=torch.float)
        l2 = torch.tensor(self.volLabels[idx], dtype=torch.float)
        feature = torch.cat([f1, f2])
        label = torch.cat([l1, l2])
        return feature, label


########################################## Load datasets ##########################################
def load_ShortTermPred(ckpt_dir: str = 'C:/Users/Home/PythonProjects/hwyTrafficPred/toolkits/cnndataset') -> tuple:
    """ Load Dataset for short-term prediction """
    trainDataset = CNNDataset(load_ckpt=True, mode='train', ckpt_dir=ckpt_dir)
    testDataset = CNNDataset(load_ckpt=True, mode='test', ckpt_dir=ckpt_dir)
    return trainDataset, testDataset


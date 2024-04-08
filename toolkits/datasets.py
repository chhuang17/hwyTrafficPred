from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import h5py


def train_test_split(speedCollection, volCollection, occCollection, laneCollection, tunnelCollection,
                     train_size=None, test_size=None, random_number=42):
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
    trainOcc = list(pd.Series(occCollection)[list(trainDataIdx)])
    trainNumLane = list(pd.Series(laneCollection)[list(trainDataIdx)])
    trainTunnel = list(pd.Series(tunnelCollection)[list(trainDataIdx)])

    testSpeed = list(pd.Series(speedCollection)[list(testDataIdx)])
    testVol = list(pd.Series(volCollection)[list(testDataIdx)])
    testOcc = list(pd.Series(occCollection)[list(testDataIdx)])
    testNumLane = list(pd.Series(laneCollection)[list(testDataIdx)])
    testTunnel = list(pd.Series(tunnelCollection)[list(testDataIdx)])

    return trainSpeed, trainVol, trainOcc, trainNumLane, trainTunnel,\
            testSpeed, testVol, testOcc, testNumLane, testTunnel

def minMaxScaler(tensor: torch.Tensor,
                 feature: str,
                 max_speed: float = 100,
                 max_volume: float = 250,
                 max_occ: float = 100,
                 max_lane: int = 4) -> torch.Tensor:
    
    if (feature == 'speed'):
        speed = torch.where(tensor>max_speed, max_speed, tensor)
        return torch.where(speed<0, -1, speed/max_speed)
    
    elif (feature == 'volume'):
        volume = torch.where(tensor>max_volume, max_volume, tensor)
        return torch.where(volume<0, -1, volume/max_volume)
    
    elif (feature == 'occupancy'):
        return torch.where(tensor<0, -1, tensor/max_occ)
    
    elif (feature == 'lane'):
        return tensor/max_lane
    
    else:
        return tensor


################################## Datasets inherited from torch.utils.data.Dataset ##################################
class ShortTermTrafficDataset(Dataset):
    def __init__(
            self,
            speed_data: list = None,
            volume_data: list = None,
            occupy_data: list = None,
            lane_data: list = None,
            tunnel_data: list = None,
            load_ckpt: bool = None,
            mode: str = None,
            ckpt_dir: str = '../hwyTrafficPred/toolkits/next_half_hour_data') -> None:
        
        if (load_ckpt):
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_speed_feature.h5", 'r') as file:
                self.speedFeature = file[f"{mode}_speed_feature"][:]
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_volume_feature.h5", 'r') as file:
                self.volFeature = file[f"{mode}_volume_feature"][:]
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_occupancy_feature.h5", 'r') as file:
                self.occFeature = file[f"{mode}_occupancy_feature"][:]
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_numlane_feature.h5", 'r') as file:
                self.laneFeature = file[f"{mode}_numlane_feature"][:]
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_tunnel_feature.h5", 'r') as file:
                self.tunnelFeature = file[f"{mode}_tunnel_feature"][:]
            
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_speed_label.h5", 'r') as file:
                self.speedLabels = file[f"{mode}_speed_label"][:]
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_volume_label.h5", 'r') as file:
                self.volLabels = file[f"{mode}_volume_label"][:]
        else:
            self.speedFeature, self.volFeature, self.occFeature, self.laneFeature, self.tunnelFeature =\
                [], [], [], [], []            
            self.speedLabels, self.volLabels = [], []
            valid_indices = [i for i, x in enumerate(speed_data) if x[1][1][0] >= 0 and volume_data[i][1][1][0] >= 0]
            
            self.speedFeature = np.array([speed_data[i][0] for i in valid_indices])
            self.volFeature = np.array([volume_data[i][0] for i in valid_indices])
            self.occFeature = np.array([occupy_data[i][0] for i in valid_indices])
            self.laneFeature = np.array([lane_data[i][0] for i in valid_indices])
            self.tunnelFeature = np.array([tunnel_data[i][0] for i in valid_indices])

            self.speedLabels = np.array([speed_data[i][1][[1],:] for i in valid_indices])
            self.volLabels = np.array([volume_data[i][1][[1],:] for i in valid_indices])

            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_speed_feature.h5", 'w') as f:
                f.create_dataset(f"{mode}_speed_feature", data=self.speedFeature)
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_volume_feature.h5", 'w') as f:
                f.create_dataset(f"{mode}_volume_feature", data=self.volFeature)
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_occupancy_feature.h5", 'w') as f:
                f.create_dataset(f"{mode}_occupancy_feature", data=self.occFeature)
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_numlane_feature.h5", 'w') as f:
                f.create_dataset(f"{mode}_numlane_feature", data=self.laneFeature)
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_tunnel_feature.h5", 'w') as f:
                f.create_dataset(f"{mode}_tunnel_feature", data=self.tunnelFeature)
            
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_speed_label.h5", 'w') as f:
                f.create_dataset(f"{mode}_speed_label", data=self.speedLabels)
            with h5py.File(f"{ckpt_dir}/{mode}/{mode}_volume_label.h5", 'w') as f:
                f.create_dataset(f"{mode}_volume_label", data=self.volLabels)

    def __len__(self) -> int:
        return len(self.speedFeature)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        f1 = minMaxScaler(torch.tensor(self.speedFeature[idx], dtype=torch.float).unsqueeze(0), feature='speed')
        f2 = minMaxScaler(torch.tensor(self.volFeature[idx], dtype=torch.float).unsqueeze(0), feature='volume')
        f3 = minMaxScaler(torch.tensor(self.occFeature[idx], dtype=torch.float).unsqueeze(0), feature='occupancy')
        f4 = minMaxScaler(torch.tensor(self.laneFeature[idx], dtype=torch.float).unsqueeze(0), feature='lane')
        f5 = torch.tensor(self.tunnelFeature[idx], dtype=torch.float).unsqueeze(0)

        l1 = minMaxScaler(torch.tensor(self.speedLabels[idx], dtype=torch.float).squeeze(0), feature='speed')
        l2 = minMaxScaler(torch.tensor(self.volLabels[idx], dtype=torch.float).squeeze(0), feature='volume')
        feature = torch.cat([f1, f2, f3, f4, f5])
        label = torch.cat([l1, l2])
        return feature, label


########################################## Load datasets ##########################################
def load_next_5min(ckpt_dir: str = '../hwyTrafficPred/toolkits/next_five_min_data') -> tuple:
    trainDataset = ShortTermTrafficDataset(load_ckpt=True, mode='train', ckpt_dir=ckpt_dir)
    testDataset = ShortTermTrafficDataset(load_ckpt=True, mode='test', ckpt_dir=ckpt_dir)
    return trainDataset, testDataset

def load_next_30min(ckpt_dir: str = '../hwyTrafficPred/toolkits/next_half_hour_data') -> tuple:
    trainDataset = ShortTermTrafficDataset(load_ckpt=True, mode='train', ckpt_dir=ckpt_dir)
    testDataset = ShortTermTrafficDataset(load_ckpt=True, mode='test', ckpt_dir=ckpt_dir)
    return trainDataset, testDataset


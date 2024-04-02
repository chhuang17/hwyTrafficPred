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

def minMaxScaler(tensor: torch.Tensor, max_speed: float = 100, max_volume: float = 240, max_occ: float = 100, max_lane: int = 4) -> torch.Tensor:
    if (len(tensor) == 5):
        # Speed
        speed = torch.where(tensor[0]>max_speed, max_speed, tensor[0])
        speed = torch.where(tensor[0]<0, -1, tensor[0]/max_speed).unsqueeze(0)

        # Volume
        volume = torch.where(tensor[1]>max_volume, max_volume, tensor[1])
        volume = torch.where(tensor[1]<0, -1, tensor[1]/max_volume).unsqueeze(0)
        
        # Occupancy
        occupy = torch.where(tensor[2]<0, -1, tensor[2]/max_occ).unsqueeze(0)

        # Number of Lane
        lanes = (tensor[3] / max_lane).unsqueeze(0)

        return torch.cat([speed, volume, occupy, lanes, tensor[-1].unsqueeze(0)])
    
    elif (len(tensor) == 2):
        speed = torch.where(tensor[0]>max_speed, 1, tensor[0]/max_speed).unsqueeze(0)
        volume = torch.where(tensor[1]>max_volume, 1, tensor[1]/max_volume).unsqueeze(0)
        return torch.cat([speed, volume])


################################## Datasets inherited from torch.utils.data.Dataset ##################################
class CNNDataset(Dataset):
    def __init__(
            self,
            speed_data: list = None,
            volume_data: list = None,
            occupy_data: list = None,
            lane_data: list = None,
            tunnel_data: list = None,
            load_ckpt: bool = None,
            mode: str = None,
            ckpt_dir: str = 'C:/Users/Home/PythonProjects/hwyTrafficPred/toolkits/cnndataset'
    ) -> None:
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
            for x in range(len(speed_data)):
                # Labels must be valid (>=0), or it will be dropped.
                if (speed_data[x][1][1][0] >= 0) and (volume_data[x][1][1][0] >= 0):                
                    self.speedFeature.append(speed_data[x][0])
                    self.volFeature.append(volume_data[x][0])
                    self.occFeature.append(occupy_data[x][0])
                    self.laneFeature.append(lane_data[x][0])
                    self.tunnelFeature.append(tunnel_data[x][0])

                    self.speedLabels.append(speed_data[x][1][[1],:])
                    self.volLabels.append(volume_data[x][1][[1],:])

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
        f1 = torch.tensor(self.speedFeature[idx], dtype=torch.float).unsqueeze(0)
        f2 = torch.tensor(self.volFeature[idx], dtype=torch.float).unsqueeze(0)
        f3 = torch.tensor(self.occFeature[idx], dtype=torch.float).unsqueeze(0)
        f4 = torch.tensor(self.laneFeature[idx], dtype=torch.float).unsqueeze(0)
        f5 = torch.tensor(self.tunnelFeature[idx], dtype=torch.float).unsqueeze(0)

        l1 = torch.tensor(self.speedLabels[idx], dtype=torch.float).squeeze(0)
        l2 = torch.tensor(self.volLabels[idx], dtype=torch.float).squeeze(0)
        feature = minMaxScaler(torch.cat([f1, f2, f3, f4, f5]))
        label = minMaxScaler(torch.cat([l1, l2]))
        return feature, label


########################################## Load datasets ##########################################
def load_next_5min(ckpt_dir: str = 'C:/Users/Home/PythonProjects/hwyTrafficPred/toolkits/cnndataset') -> tuple:
    """ Load Dataset for short-term prediction """
    trainDataset = CNNDataset(load_ckpt=True, mode='train', ckpt_dir=ckpt_dir)
    testDataset = CNNDataset(load_ckpt=True, mode='test', ckpt_dir=ckpt_dir)
    return trainDataset, testDataset


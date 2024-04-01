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

def min_max_scaler(arr: np.ndarray, feature: str) -> np.ndarray:
    if (feature == 'speed') or (feature == 'occ'):
        arr = np.where(arr>=100, 100, arr)
        return np.where(arr<0, -1, arr/100)
    elif (feature == 'volume'):
        arr = np.where(arr>=600, 600, arr)
        return np.where(arr<0, -1, arr/600)
    else:
        raise ValueError(f"'{feature}'")


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
            # with h5py.File(f"{ckpt_dir}/{mode}/{mode}_occupancy_label.h5", 'r') as file:
            #     self.occLabels = file[f"{mode}_occupancy_label"][:]
            # with h5py.File(f"{ckpt_dir}/{mode}/{mode}_numlane_label.h5", 'r') as file:
            #     self.laneLabels = file[f"{mode}_numlane_label"][:]
            # with h5py.File(f"{ckpt_dir}/{mode}/{mode}_tunnel_label.h5", 'r') as file:
            #     self.tunnelLabels = file[f"{mode}_tunnel_label"][:]
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
                    # self.occLabels.append(occupy_data[x][1][[1],:])
                    # self.laneLabels.append(lane_data[x][1][[1],:])
                    # self.tunnelLabels.append(tunnel_data[x][1][[1],:])

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
            # with h5py.File(f"{ckpt_dir}/{mode}/{mode}_occupancy_label.h5", 'w') as f:
            #     f.create_dataset(f"{mode}_occupancy_label", data=self.occLabels)
            # with h5py.File(f"{ckpt_dir}/{mode}/{mode}_numlane_label.h5", 'w') as f:
            #     f.create_dataset(f"{mode}_numlane_label", data=self.laneLabels)
            # with h5py.File(f"{ckpt_dir}/{mode}/{mode}_tunnel_label.h5", 'w') as f:
            #     f.create_dataset(f"{mode}_tunnel_label", data=self.tunnelLabels)

    def __len__(self) -> int:
        return len(self.speedFeature)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        f1 = min_max_scaler(self.speedFeature[idx], 'speed')
        f2 = min_max_scaler(self.volFeature[idx], 'volume')
        f3 = min_max_scaler(self.occFeature[idx], 'occ')
        f4 = self.laneFeature[idx]
        f5 = self.tunnelFeature[idx]
        
        l1 = min_max_scaler(self.speedLabels[idx], 'speed')
        l2 = min_max_scaler(self.volLabels[idx], 'volume')

        f1 = torch.tensor(f1, dtype=torch.float).unsqueeze(0)
        f2 = torch.tensor(f2, dtype=torch.float).unsqueeze(0)
        f3 = torch.tensor(f3, dtype=torch.float).unsqueeze(0)
        f4 = torch.tensor(f4, dtype=torch.float).unsqueeze(0)
        f5 = torch.tensor(f5, dtype=torch.float).unsqueeze(0)

        l1 = torch.tensor(l1, dtype=torch.float).squeeze(0)
        l2 = torch.tensor(l2, dtype=torch.float).squeeze(0)
        feature = torch.cat([f1, f2, f3, f4, f5])
        label = torch.cat([l1, l2])
        return feature, label


########################################## Load datasets ##########################################
def load_next_5min(ckpt_dir: str = 'C:/Users/Home/PythonProjects/hwyTrafficPred/toolkits/cnndataset') -> tuple:
    """ Load Dataset for short-term prediction """
    trainDataset = CNNDataset(load_ckpt=True, mode='train', ckpt_dir=ckpt_dir)
    testDataset = CNNDataset(load_ckpt=True, mode='test', ckpt_dir=ckpt_dir)
    return trainDataset, testDataset


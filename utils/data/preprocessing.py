import pandas as pd
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from typing import Tuple, List


def train_test_split(speeds: List,
                     vols: List,
                     occs: List,
                     lanes: List,
                     tunnels: List,
                     train_size: float = None,
                     test_size: float = None,
                     random_number: int = 42
                     ) -> Tuple[List, List, List, List, List,
                                List, List, List, List, List]:
    np.random.seed(random_number)
    if (train_size):
        train_indices = np.random.choice(
            len(speeds),
            int(train_size * len(speeds)),
            replace=False
        )
        test_indices = set([i for i in range(len(speeds))]) -\
                       set(train_indices)
    
    elif (test_size):
        test_indices = np.random.choice(
            len(speeds),
            int(test_size * len(speeds)),
            replace=False
        )
        train_indices = set([i for i in range(len(speeds))]) -\
                        set(test_indices)
        
    train_speeds = list(pd.Series(speeds)[list(train_indices)])
    train_vols = list(pd.Series(vols)[list(train_indices)])
    train_occs = list(pd.Series(occs)[list(train_indices)])
    train_lanes = list(pd.Series(lanes)[list(train_indices)])
    train_tunnels = list(pd.Series(tunnels)[list(train_indices)])

    test_speeds = list(pd.Series(speeds)[list(test_indices)])
    test_vols = list(pd.Series(vols)[list(test_indices)])
    test_occs = list(pd.Series(occs)[list(test_indices)])
    test_lanes = list(pd.Series(lanes)[list(test_indices)])
    test_tunnels = list(pd.Series(tunnels)[list(test_indices)])

    return train_speeds, train_vols, train_occs, train_lanes, train_tunnels,\
           test_speeds, test_vols, test_occs, test_lanes, test_tunnels

def min_max_scaler(tensor: torch.Tensor,
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
    def __init__(self,
                 speed_data: List = None,
                 volume_data: List = None,
                 occupy_data: List = None,
                 lane_data: List = None,
                 tunnel_data: List = None,
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
        # features
        f1 = min_max_scaler(
            tensor=torch.tensor(
                self.speedFeature[idx], dtype=torch.float
            ).unsqueeze(0),
            feature='speed'
        )
        f2 = min_max_scaler(
            tensor=torch.tensor(
                self.volFeature[idx], dtype=torch.float
            ).unsqueeze(0),
            feature='volume'
        )
        f3 = min_max_scaler(
            tensor=torch.tensor(
                self.occFeature[idx], dtype=torch.float
            ).unsqueeze(0),
            feature='occupancy'
        )
        f4 = min_max_scaler(
            tensor=torch.tensor(
                self.laneFeature[idx], dtype=torch.float
            ).unsqueeze(0),
            feature='lane'
        )
        f5 = torch.tensor(self.tunnelFeature[idx],
                          dtype=torch.float).unsqueeze(0)

        # label
        l1 = min_max_scaler(
            tensor=torch.tensor(
                self.speedLabels[idx],dtype=torch.float
            ).squeeze(0),
            feature='speed'
        )
        l2 = min_max_scaler(
            tensor=torch.tensor(
                self.volLabels[idx], dtype=torch.float
            ).squeeze(0),
            feature='volume'
        )

        return torch.cat([f1, f2, f3, f4, f5]), torch.cat([l1, l2])


########################################## Load datasets ##########################################
def load_next_5min(ckpt_dir: str = '../hwyTrafficPred/toolkits/next_five_min_data') -> Tuple:
    trainDataset = ShortTermTrafficDataset(load_ckpt=True, mode='train', ckpt_dir=ckpt_dir)
    testDataset = ShortTermTrafficDataset(load_ckpt=True, mode='test', ckpt_dir=ckpt_dir)
    return trainDataset, testDataset

def load_next_30min(ckpt_dir: str = '../hwyTrafficPred/toolkits/next_half_hour_data') -> Tuple:
    trainDataset = ShortTermTrafficDataset(load_ckpt=True, mode='train', ckpt_dir=ckpt_dir)
    testDataset = ShortTermTrafficDataset(load_ckpt=True, mode='test', ckpt_dir=ckpt_dir)
    return trainDataset, testDataset


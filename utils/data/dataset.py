import sys
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from typing import List

sys.path.append('/app')
from utils.data.preprocessing import min_max_scaler


class TimeSeriesTrafficDataset(Dataset):
    def __init__(self,
                 speed_data: List = None,
                 volume_data: List = None,
                 occupy_data: List = None,
                 lane_data: List = None,
                 tunnel_data: List = None,
                 load_ckpt: bool = False,
                 usage: str = 'train',
                 ckpt_dir: str = '../hwyTrafficPred/toolkits/next_half_hour_data'
                 ) -> None:
        
        if (load_ckpt):
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_speed_feature.h5",
                           mode='r') as file:
                self.feature_speed = file[f"{usage}_speed_feature"][:]
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_volume_feature.h5",
                           mode='r') as file:
                self.feature_volume = file[f"{usage}_volume_feature"][:]
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_occupancy_feature.h5",
                           mode='r') as file:
                self.feature_occupy = file[f"{usage}_occupancy_feature"][:]
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_numlane_feature.h5",
                           mode='r') as file:
                self.feature_lane = file[f"{usage}_numlane_feature"][:]
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_tunnel_feature.h5",
                           mode='r') as file:
                self.feature_tunnel = file[f"{usage}_tunnel_feature"][:]
            
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_speed_label.h5",
                           mode='r') as file:
                self.label_speed = file[f"{usage}_speed_label"][:]
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_volume_label.h5",
                           mode='r') as file:
                self.label_volume = file[f"{usage}_volume_label"][:]
        else:
            self.feature_speed, self.feature_volume, self.feature_occupy,\
                self.feature_lane, self.feature_tunnel = [], [], [], [], []            
            
            self.label_speed, self.label_volume = [], []
            
            valid_indices = [
                i for i, x in enumerate(speed_data)
                if x[1][1][0] >= 0 and volume_data[i][1][1][0] >= 0
            ]
            
            self.feature_speed = np.array([
                speed_data[i][0] for i in valid_indices
            ])
            self.feature_volume = np.array([
                volume_data[i][0] for i in valid_indices
            ])
            self.feature_occupy = np.array([
                occupy_data[i][0] for i in valid_indices
            ])
            self.feature_lane = np.array([
                lane_data[i][0] for i in valid_indices
            ])
            self.feature_tunnel = np.array([
                tunnel_data[i][0] for i in valid_indices
            ])

            self.label_speed = np.array([
                speed_data[i][1][[1],:] for i in valid_indices
            ])
            self.label_volume = np.array([
                volume_data[i][1][[1],:] for i in valid_indices
            ])
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_speed_feature.h5",
                           mode='w') as f:
                f.create_dataset(f"{usage}_speed_feature",
                                 data=self.feature_speed)
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_volume_feature.h5",
                           mode='w') as f:
                f.create_dataset(f"{usage}_volume_feature",
                                 data=self.feature_volume)
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_occupancy_feature.h5",
                           mode='w') as f:
                f.create_dataset(f"{usage}_occupancy_feature",
                                 data=self.feature_occupy)
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_numlane_feature.h5",
                           mode='w') as f:
                f.create_dataset(f"{usage}_numlane_feature",
                                 data=self.feature_lane)
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_tunnel_feature.h5",
                           mode='w') as f:
                f.create_dataset(f"{usage}_tunnel_feature",
                                 data=self.feature_tunnel)
            
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_speed_label.h5",
                           mode='w') as f:
                f.create_dataset(f"{usage}_speed_label",
                                 data=self.label_speed)
            
            with h5py.File(name=f"{ckpt_dir}/{usage}/{usage}_volume_label.h5",
                           mode='w') as f:
                f.create_dataset(f"{usage}_volume_label",
                                 data=self.label_volume)

    def __len__(self) -> int:
        return len(self.feature_speed)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # features
        f1 = min_max_scaler(
            tensor=torch.tensor(
                self.feature_speed[idx], dtype=torch.float
            ).unsqueeze(0),
            feature='speed'
        )
        f2 = min_max_scaler(
            tensor=torch.tensor(
                self.feature_volume[idx], dtype=torch.float
            ).unsqueeze(0),
            feature='volume'
        )
        f3 = min_max_scaler(
            tensor=torch.tensor(
                self.feature_occupy[idx], dtype=torch.float
            ).unsqueeze(0),
            feature='occupancy'
        )
        f4 = min_max_scaler(
            tensor=torch.tensor(
                self.feature_lane[idx], dtype=torch.float
            ).unsqueeze(0),
            feature='lane'
        )
        f5 = torch.tensor(self.feature_tunnel[idx],
                          dtype=torch.float).unsqueeze(0)

        # label
        l1 = min_max_scaler(
            tensor=torch.tensor(
                self.label_speed[idx],dtype=torch.float
            ).squeeze(0),
            feature='speed'
        )
        l2 = min_max_scaler(
            tensor=torch.tensor(
                self.label_volume[idx], dtype=torch.float
            ).squeeze(0),
            feature='volume'
        )

        return torch.cat([f1, f2, f3, f4, f5]), torch.cat([l1, l2])

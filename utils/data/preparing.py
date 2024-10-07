import pandas as pd
import numpy as np
import feather
import os
from sqlalchemy import create_engine
from datetime import datetime
from typing import List, Dict, Tuple


DB_USER = "root"
DB_PSWD = "Curry5566"
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "fwy_n5"
DB_ENGINE = "mysql+pymysql://root:Curry5566@127.0.0.1:3306/fwy_n5?charset=utf8"
engine = create_engine(DB_ENGINE)

def query_raw_dataframe(date: str) -> pd.DataFrame:
    r"""Query the daily dynamic data collected by vehicle detetors.
        Args:
            date (str): The date which we are interested in

        Returns:
            output (DataFrame): Daily dynamic data collected by vehicle detetors
    """
    sql  = " SELECT "
    sql += " 	STAC.VDID, STAC.RoadName, STAC.`Start`, STAC.`End`, STAC.RoadDirection, "
    sql += "    CASE "
    sql += "        WHEN DYMC.Occupancy = 0 AND DYMC.Volume = 0 THEN 100 "
    sql += "        ELSE DYMC.Speed "
    sql += " 	END AS Speed, "
    sql += "    DYMC.Occupancy, DYMC.Volume, "
    sql += " 	STAC.ActualLaneNum, STAC.LocationMile, STAC.isTunnel, DYMC.DataCollectTime "
    sql += " FROM ( "
    sql += " 	SELECT "
    sql += " 		VDSTC.id, VDSTC.VDID, ROAD.RoadName, SEC.`Start`, SEC.`End`, "
    sql += " 		VDSTC.ActualLaneNum, VDSTC.RoadDirection, VDSTC.LocationMile, "
    sql += "        CASE "
    sql += " 	        WHEN VDSTC.RoadDirection = 'S' AND VDSTC.LocationMile BETWEEN 0.238 AND 0.694 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'N' AND VDSTC.LocationMile BETWEEN 0.235 AND 0.690 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'S' AND VDSTC.LocationMile BETWEEN 0.694 AND 3.481 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'N' AND VDSTC.LocationMile BETWEEN 0.795 AND 3.515 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'S' AND VDSTC.LocationMile BETWEEN 7.677 AND 7.893 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'N' AND VDSTC.LocationMile BETWEEN 7.646 AND 7.894 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'S' AND VDSTC.LocationMile BETWEEN 9.442 AND 13.303 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'N' AND VDSTC.LocationMile BETWEEN 9.457 AND 13.263 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'S' AND VDSTC.LocationMile BETWEEN 15.203 AND 28.128 THEN 1 "
    sql += " 	        WHEN VDSTC.RoadDirection = 'N' AND VDSTC.LocationMile BETWEEN 15.179 AND 28.134 THEN 1 "
    sql += " 	        ELSE 0 "
    sql += "        END AS isTunnel "
    sql += " 	FROM fwy_n5.vd_static_2023 VDSTC "
    sql += " 	JOIN transport.road_info ROAD ON VDSTC.RoadInfoID = ROAD.id "
    sql += " 	JOIN transport.section_info SEC ON ROAD.id = SEC.RoadInfoID "
    sql += " 	AND VDSTC.LocationMile >= SEC.StartKM "
    sql += " 	AND VDSTC.LocationMile <= SEC.EndKM "
    sql += " 	WHERE VDSTC.Mainlane = 1 "
    sql += " ) STAC JOIN ( "
    sql += " 	SELECT "
    sql += " 		VdStaticID, "
    sql += "        COUNT(VdStaticID), "
    sql += " 		CASE "
    sql += " 			WHEN MIN(Speed) = -99 THEN -99 "
    sql += " 			ELSE AVG(Speed) "
    sql += " 		END AS Speed,  "
    sql += " 		CASE "
    sql += " 			WHEN MIN(Occupancy) = -99 THEN -99 "
    sql += " 			ELSE AVG(Occupancy) "
    sql += " 		END AS Occupancy,  "
    sql += " 		CASE "
    sql += " 			WHEN MIN(Volume) = -99 THEN -99 "
    sql += " 			ELSE SUM(Volume) "
    sql += " 		END AS Volume, "
    sql += " 		MAX(DataCollectTime) AS DataCollectTime, "
    sql += " 		(UNIX_TIMESTAMP(DataCollectTime)-UNIX_TIMESTAMP(%(date)s)) DIV 300 "
    sql += " 	FROM fwy_n5.vd_dynamic_detail_{} ".format(date.replace('-',''))
    sql += " 	GROUP BY VdStaticID, (UNIX_TIMESTAMP(DataCollectTime)-UNIX_TIMESTAMP(%(date)s)) DIV 300 "
    sql += "    HAVING COUNT(VdStaticID) %% 5 = 0 "
    sql += " ) DYMC ON STAC.id = DYMC.VdStaticID "
    sql += " ORDER BY STAC.RoadDirection, STAC.LocationMile, DYMC.DataCollectTime; "

    df = pd.read_sql(sql, con=engine, params={'date': date})
    engine.dispose()
    return df.sort_values(
        by=['RoadDirection','DataCollectTime','LocationMile']
    ).reset_index(drop=True)

def generate_vd_groups(df: pd.DataFrame, group_size: int) -> Dict[str, List]:
    r"""Group all vehicle detectors.
        Args:
            df (DataFrame): Reference table
            group_size (int): The size of the VD group

        Returns:
            output (Dict[str, List]): The group dict of vehicle detectors
    """
    vd_group_dict = {}
    lb = group_size // 2
    ub = group_size - (group_size // 2)
    for vdid in df['VDID'].unique():
        vd_group_dict.setdefault(f"{vdid}", [])
    for no, vdid in enumerate(df['VDID'].unique()):
        start_idx = max(no-lb, 0)
        end_idx = min(no+ub, df['VDID'].nunique())
        vd_group_dict[f"{vdid}"] += list(df['VDID'].unique()[start_idx:no]) +\
                                    list(df['VDID'].unique()[no:end_idx])

    delete_list = []
    for k in vd_group_dict.keys():
        if (len(vd_group_dict[k]) != group_size):
            delete_list.append(k)
    for k in delete_list:
        del vd_group_dict[k]
    
    return vd_group_dict

def generate_samples(df: pd.DataFrame,
                     vd_group_dict: dict,
                     target_vdid: str,
                     group_size: int,
                     input_duration: int = 30,
                     output_duration: int = 30
                     ) -> Tuple[List[Tuple], List[Tuple], List[Tuple],
                                List[Tuple], List[Tuple]]:
    r"""Generate samples from the given dataframe.

        Args:
            df (DataFrame): Reference table
            vd_group_dict (Dict): The group dict of vehicle detectors, which is
                                  the output of :func:`generate_vd_groups()`
            target_vdid (str): The VDID which we are interested in and is used
                               as a key of :attr:`vd_group_dict`
            group_size (int): The size of the VD group
            input_duration (int, optional): Duration for the input window (minutes).
                                            Defaults to 30
            output_duration (int, optional): Duration for the output window (minutes).
                                             Defaults to 30

        Returns:
            output (Tuple[List[Tuple], List[Tuple], List[Tuple], List[Tuple], List[Tuple]]):
            Tuple which is consists of lists.
            - **speeds_samples**: Position at first and consists of tuples,
                                  each tuple is regarded as **(X,y)**
            - **vol_samples**: Position at second and consists of tuples,
                               each tuple is regarded as **(X,y)**
            - **occ_samples**: Position at third and consists of tuples,
                               each tuple is regarded as **(X,y)**
            - **lane_samples**: Position at fourth and consists of tuples,
                                each tuple is regarded as **(X,y)**
            - **tunnel_samples**: Position at fifth and consists of tuples,
                                  each tuple is regarded as **(X,y)**
    """
    speed_samples, vol_samples, occ_samples, lane_samples, tunnel_samples =\
        [], [], [], [], []
    
    vd_group_df = df.loc[
        (df['VDID'].isin(vd_group_dict[f"{target_vdid}"]))
    ].sort_values(by=['LocationMile', 'DataCollectTime'])

    indices = [x for x in range(
        0, vd_group_df.shape[0]+1, vd_group_df.shape[0]//group_size
    )]
    speed_arr = np.zeros((group_size, vd_group_df.shape[0]//group_size))
    vol_arr = np.zeros((group_size, vd_group_df.shape[0]//group_size))
    occ_arr = np.zeros((group_size, vd_group_df.shape[0]//group_size))
    lane_arr = np.zeros((group_size, vd_group_df.shape[0]//group_size))
    tunnel_arr = np.zeros((group_size, vd_group_df.shape[0]//group_size))
    for i, j, k in zip(range(group_size), indices[:-1], indices[1:]):
        speed_arr[i] += vd_group_df.iloc[j:k,:]['Speed'].to_numpy()
        vol_arr[i] += vd_group_df.iloc[j:k,:]['Volume'].to_numpy()
        occ_arr[i] += vd_group_df.iloc[j:k,:]['Occupancy'].to_numpy()
        lane_arr[i] += vd_group_df.iloc[j:k,:]['ActualLaneNum'].to_numpy()
        tunnel_arr[i] += vd_group_df.iloc[j:k,:]['isTunnel'].to_numpy()

    input_length = input_duration//5
    output_length = output_duration//5
    slice_length = input_length + output_length
    
    for x in range(speed_arr.shape[1]//slice_length*slice_length-(slice_length-1)):
        if (-99 not in speed_arr[:,x:x+slice_length][:,-1*output_length:][1,:]):
            speed_samples.append(
                (speed_arr[:,x:x+slice_length][:,:input_length],
                 speed_arr[:,x:x+slice_length][:,-1*output_length:])
            )
            vol_samples.append(
                (vol_arr[:,x:x+slice_length][:,:input_length],
                 vol_arr[:,x:x+slice_length][:,-1*output_length:])
            )
            occ_samples.append(
                (occ_arr[:,x:x+slice_length][:,:input_length],
                 occ_arr[:,x:x+slice_length][:,-1*output_length:])
            )
            lane_samples.append(
                (lane_arr[:,x:x+slice_length][:,:input_length],
                 lane_arr[:,x:x+slice_length][:,-1*output_length:])
            )
            tunnel_samples.append(
                (tunnel_arr[:,x:x+slice_length][:,:input_length],
                 tunnel_arr[:,x:x+slice_length][:,-1*output_length:])
            )
    
    return speed_samples, vol_samples, occ_samples,\
           lane_samples, tunnel_samples

def download_monthly_tables(start: str,
                            end: str,
                            dest_dir: str,
                            file_format: str = 'feather') -> None:
    """ Download monthly tables from db
        ```text
        ---
        @Params
        df: 
        start: The start date for querying.
        end: The end date for querying.
        dest_dir: Directory for files saved to.
        file_format: The table(s) would be saved as which type.

        ---
        @Returns
        None
        ```
    """
    start_date = list(map(lambda x: datetime.strftime(x, '%Y-%m-%d'),
                          list(pd.date_range(start, end, freq='MS'))))
    end_date = list(map(lambda x: datetime.strftime(x, '%Y-%m-%d'),
                        list(pd.date_range(start, end, freq='ME'))))

    for start, end in zip(start_date, end_date):
        dataframes = []
        dateList = list(map(lambda x: datetime.strftime(x, '%Y-%m-%d'),
                            list(pd.date_range(start, end))))
        for date in dateList:
            dataframes.append(query_raw_dataframe(date))
        dataframes = pd.concat(dataframes).reset_index(drop=True)
        
        if (file_format == 'feather'):
            feather.write_dataframe(
                dataframes,
                dest=f"{dest_dir}/{start[:7].replace('-','')}.feather"
            )
        elif (file_format == 'csv'):
            dataframes.to_csv(
                f"{dest_dir}/{start[:7].replace('-','')}.csv",
                index=False,
                encoding='utf_8_sig'
            )
        else:
            raise ValueError(
                f"'{file_format}' is not supported, choose feather or csv."
            )
        
def concat_tables(folder: str = './nfb2023',
                  file_format: str = 'feather') -> pd.DataFrame:
    df = []
    for filename in os.listdir(folder):
        if (file_format == 'feather'):
            monthly_df = feather.read_dataframe(f"{folder}/{filename}")
        elif (file_format == 'csv'):
            monthly_df = pd.read_csv(f"{folder}/{filename}")
        else:
            raise ValueError(
                f"'{file_format}' is not supported, choose feather or csv."
            )
        
        if (len(df) == 0):
            df.append(monthly_df)
        else:
            current_df = pd.concat(df).reset_index(drop=True)
            monthly_df = monthly_df.loc[
                monthly_df['VDID'].isin(set(current_df['VDID']))
            ]
            df.append(monthly_df)

    df = pd.concat(df)
    
    # Remove outliers
    df = df.loc[(df['Speed']<=120) &\
                (df['Volume']<=3200/12)].reset_index(drop=True)    
    return df

def collect_samples(group_size: int = 3,
                    data_dir: str = './nfb2023',
                    file_format: str = 'feather',
                    input_duration: int = 30,
                    output_duration: int = 30
                    ) -> Tuple[List[Tuple], List[Tuple], List[Tuple],
                               List[Tuple], List[Tuple]]:
    
    df = concat_tables(folder=data_dir, file_format=file_format)
    speeds, vols, occs, lanes, tunnels = [], [], [], [], []

    # Northbound data
    northbound_df = df.loc[df['RoadDirection']=='N'].reset_index(drop=True)
    northbound_vd_groups = generate_vd_groups(df=northbound_df,
                                              group_size=group_size)
    for target_vdid in northbound_vd_groups.keys():
        speed_samples, vol_samples, occ_samples, lane_samples, tunnel_samples =\
            generate_samples(
                df=northbound_df,
                vd_group_dict=northbound_vd_groups,
                target_vdid=target_vdid,
                group_size=group_size,
                input_duration=input_duration,
                output_duration=output_duration
            )
        speeds.extend(speed_samples)
        vols.extend(vol_samples)
        occs.extend(occ_samples)
        lanes.extend(lane_samples)
        tunnels.extend(tunnel_samples)

    # Southbound data
    southbound_df = df.loc[df['RoadDirection']=='S'].reset_index(drop=True)
    southbound_vd_groups = generate_vd_groups(df=southbound_df,
                                              group_size=group_size)
    for target_vdid in southbound_vd_groups.keys():
        speed_samples, vol_samples, occ_samples, lane_samples, tunnel_samples =\
            generate_samples(
                df=southbound_df,
                vd_group_dict=southbound_vd_groups,
                target_vdid=target_vdid,
                group_size=group_size,
                input_duration=input_duration,
                output_duration=output_duration
            )
        speeds.extend(speed_samples)
        vols.extend(vol_samples)
        occs.extend(occ_samples)
        lanes.extend(lane_samples)
        tunnels.extend(tunnel_samples)

    return speeds, vols, occs, lanes, tunnels

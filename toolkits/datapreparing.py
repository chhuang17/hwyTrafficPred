from sqlalchemy import create_engine
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import feather
import os


# db config
load_dotenv('./.env')
engine = create_engine(os.getenv('DB_ENGINE'))

def getRollingMeanDaily(selectDate: str) -> pd.DataFrame:
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
    sql += " 		(UNIX_TIMESTAMP(DataCollectTime)-UNIX_TIMESTAMP(%(selectDate)s)) DIV 300 "
    sql += " 	FROM fwy_n5.vd_dynamic_detail_{} ".format(selectDate.replace('-',''))
    sql += " 	GROUP BY VdStaticID, (UNIX_TIMESTAMP(DataCollectTime)-UNIX_TIMESTAMP(%(selectDate)s)) DIV 300 "
    sql += "    HAVING COUNT(VdStaticID) %% 5 = 0 "
    sql += " ) DYMC ON STAC.id = DYMC.VdStaticID "
    sql += " ORDER BY STAC.RoadDirection, STAC.LocationMile, DYMC.DataCollectTime; "

    df = pd.read_sql(sql, con=engine, params={'selectDate': selectDate})
    engine.dispose()
    return df.sort_values(by=['RoadDirection','DataCollectTime','LocationMile']).reset_index(drop=True)

def groupVDs(df: pd.DataFrame, each: int) -> dict:
    """ Get the dict of VD groups
        ```text
        ---
        @Params
        df: DataFrame which is referenced by.
        each: The quantity of VDs would be considered as a group.

        ---
        @Returns
        vdGroups: The keys are the VDs we focus on, and the values are the collections of VDs which are correlated corresponding to the keys.
        ```
    """
    vdGroups = {}
    lb = each // 2
    ub = each - (each // 2)
    for vdid in df['VDID'].unique():
        vdGroups.setdefault(f"{vdid}", [])
    for no, vdid in enumerate(df['VDID'].unique()):
        startIdx = max(no-lb, 0)
        endIdx = min(no+ub, len(df['VDID'].unique())-1)
        vdGroups[f"{vdid}"] += list(df['VDID'].unique()[startIdx:no]) + list(df['VDID'].unique()[no:endIdx])

    delList = []
    for k in vdGroups.keys():
        if (len(vdGroups[k]) != each):
            delList.append(k)
    for k in delList:
        del vdGroups[k]
    
    return vdGroups

def genSamples(df: pd.DataFrame, vdGroups: dict, groupKey: str, each: int, timeWindow: int = 30) -> tuple:
    """ Generate samples for each traffic data (speed, volume, and occupancy)
        ```text
        ---
        @Params
        df: 
        vdGroups: The outpur of groupVDs(),
        groupKey: The key of vdGroups,
        each: The quantity of VDs would be considered as a group,
        timeWindow: The length of period we consider, and the default value is 30 (minutes).

        ---
        @Returns
        speeds: list with each item as a tuple, all of them are represented (X,y).
        vols: list with each item as a tuple, all of them are represented (X,y).
        occs: list with each item as a tuple, all of them are represented (X,y).
        ```
    """
    speeds, vols, occs, lanes, tunnels = [], [], [], [], []
    tmpDf = df.loc[(df['VDID'].isin(vdGroups[f"{groupKey}"]))].sort_values(by=['LocationMile', 'DataCollectTime'])

    indices = [x for x in range(0, tmpDf.shape[0]+1, tmpDf.shape[0]//each)]
    speedMatx = np.zeros((each, tmpDf.shape[0]//each))
    volMatx = np.zeros((each, tmpDf.shape[0]//each))
    occMatx = np.zeros((each, tmpDf.shape[0]//each))
    laneMatx = np.zeros((each, tmpDf.shape[0]//each))
    tunnelMatx = np.zeros((each, tmpDf.shape[0]//each))
    for i, j, k in zip(range(each), indices[:-1], indices[1:]):
        speedMatx[i] += tmpDf.iloc[j:k,:]['Speed'].to_numpy()
        volMatx[i] += tmpDf.iloc[j:k,:]['Volume'].to_numpy()
        occMatx[i] += tmpDf.iloc[j:k,:]['Occupancy'].to_numpy()
        laneMatx[i] += tmpDf.iloc[j:k,:]['ActualLaneNum'].to_numpy()
        tunnelMatx[i] += tmpDf.iloc[j:k,:]['isTunnel'].to_numpy()

    sliceLen = int((timeWindow / 5) + 1)
    for x in range(speedMatx.shape[1]//sliceLen*sliceLen-(sliceLen-1)):
        speeds.append((speedMatx[:,x:x+sliceLen][:,:-1], speedMatx[:,x:x+sliceLen][:,[-1]]))
        vols.append((volMatx[:,x:x+sliceLen][:,:-1], volMatx[:,x:x+sliceLen][:,[-1]]))
        occs.append((occMatx[:,x:x+sliceLen][:,:-1], occMatx[:,x:x+sliceLen][:,[-1]]))
        lanes.append((laneMatx[:,x:x+sliceLen][:,:-1], laneMatx[:,x:x+sliceLen][:,[-1]]))
        tunnels.append((tunnelMatx[:,x:x+sliceLen][:,:-1], tunnelMatx[:,x:x+sliceLen][:,[-1]]))
    
    return speeds, vols, occs, lanes, tunnels

def download_monthly_tables(start: str, end: str, dest_dir: str, file_format: str = 'feather') -> None:
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
    monthlyStarts = list(map(lambda x: datetime.strftime(x, '%Y-%m-%d'), list(pd.date_range(start, end, freq='MS'))))
    monthlyEnds = list(map(lambda x: datetime.strftime(x, '%Y-%m-%d'), list(pd.date_range(start, end, freq='ME'))))

    for start, end in zip(monthlyStarts, monthlyEnds):
        dataframes = []
        dateList = list(map(lambda x: datetime.strftime(x, '%Y-%m-%d'), list(pd.date_range(start, end))))
        for date in dateList:
            dataframes.append(getRollingMeanDaily(date))
        dataframes = pd.concat(dataframes).reset_index(drop=True)
        
        if (file_format == 'feather'):
            feather.write_dataframe(dataframes, dest=f"{dest_dir}/{start[:7].replace('-','')}.feather")
        elif (file_format == 'csv'):
            dataframes.to_csv(f"{dest_dir}/{start[:7].replace('-','')}.csv", index=False, encoding='utf_8_sig')
        else:
            raise ValueError(f"'{file_format}' is not supported, choose feather or csv.")
        
def concat_tables(folder: str = './nfb2023', file_format: str = 'feather'):
    df = []
    for filename in os.listdir(folder):
        if (file_format == 'feather'):
            monthlyDf = feather.read_dataframe(f"{folder}/{filename}")
        elif (file_format == 'csv'):
            monthlyDf = pd.read_csv(f"{folder}/{filename}")
        else:
            raise ValueError(f"'{file_format}' is not supported, choose feather or csv.")
        
        if (len(df) == 0):
            df.append(monthlyDf)
        else:
            currDf = pd.concat(df).reset_index(drop=True)
            monthlyDf = monthlyDf.loc[monthlyDf['VDID'].isin(set(currDf['VDID']))]
            df.append(monthlyDf)

    df = pd.concat(df)
    df = df.loc[(df['Speed']<=120) & (df['Volume']<=3200/12)].reset_index(drop=True)    
    return df

def collect_data(each_group: int = 3, data_dir: str = './nfb2023', file_format: str = 'feather'):
    df = concat_tables(folder=data_dir, file_format=file_format)
    
    speedCollection, volCollection, occCollection, laneCollection, tunnelCollection = [], [], [], [], []

    # Northbound data
    northDf = df.loc[df['RoadDirection']=='N'].reset_index(drop=True)
    northVDGrps = groupVDs(northDf, each=each_group)
    for groupKey in northVDGrps.keys():
        speeds, vols, occs, lanes, tunnels = genSamples(northDf, northVDGrps, groupKey, each=each_group, timeWindow=30)
        speedCollection += speeds
        volCollection += vols
        occCollection += occs
        laneCollection += lanes
        tunnelCollection += tunnels

    # Southbound data
    southDf = df.loc[df['RoadDirection']=='S'].reset_index(drop=True)
    southVDGrps = groupVDs(southDf, each=each_group)
    for groupKey in southVDGrps.keys():
        speeds, vols, occs, lanes, tunnels = genSamples(southDf, southVDGrps, groupKey, each=each_group, timeWindow=30)
        speedCollection += speeds
        volCollection += vols
        occCollection += occs
        laneCollection += lanes
        tunnelCollection += tunnels

    return speedCollection, volCollection, occCollection, laneCollection, tunnelCollection


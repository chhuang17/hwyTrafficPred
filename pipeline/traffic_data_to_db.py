from sqlalchemy import create_engine, MetaData, text, insert
from datetime import datetime, timedelta
from tisvcloud import vd
from dotenv import load_dotenv
import pandas as pd
import logging
import xml.etree.ElementTree as ET
import gzip
import os
import shutil
import re


logging.basicConfig(
    filename='../hwyTrafficPred/logs/traffic_data_to_db.log',
    # encoding='utf-8',
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# db config
os.environ.clear()
load_dotenv('../hwyTrafficPred/pipeline/.env')

# Force to modify the env values
os.environ['DB_NAME'] = os.getenv('DB_NAME')
os.environ['DB_ENGINE'] = os.getenv('DB_ENGINE')
engine = create_engine(os.getenv('DB_ENGINE'))


def xmlParser(filename: str) -> tuple:
    """ xml 解析器 """
    if (os.path.splitext(filename)[1] == '.xml'):
        tree = ET.parse(filename)
        root = tree.getroot()
    elif (os.path.splitext(filename)[1] == '.gz'):
        with gzip.open(filename, 'rb') as f:
            tree = ET.parse(f)
            root = tree.getroot()
    else:
        raise ValueError('File cannot be parsed.')

    # 移除命名空間前綴
    for elem in root.iter():
        if ('}' in elem.tag):
            elem.tag = elem.tag.split('}')[1]
    return tree, root

def mileageTransform(mileage: str) -> float:
    """ 里程表示方式轉換成 float """
    return round(int(mileage[:mileage.find('K')]) + float(mileage[mileage.find('+')+1:])/1000, 3)

def getNFBSectionData(filename: str) -> pd.DataFrame:
    """ 取得國道路段基本資料 """
    _, root = xmlParser(filename)
    nfb_section = []
    secs = root[-1]
    for sec in secs:
        data = {}
        for col in sec:
            if (col.tag == 'SectionID'):
                data[col.tag] = col.text
            
            if (col.tag == 'RoadSection'):
                for subcol in col:
                    data[subcol.tag] = subcol.text
            elif (col.tag == 'SectionMile'):
                for subcol in col:
                    data[subcol.tag] = mileageTransform(subcol.text)
            else:
                data[col.tag] = col.text

            data['UpdateTime'] = datetime.strptime(root[0].text, '%Y-%m-%dT%H:%M:%S+08:00')
            data['CreateTime'] = datetime.now().replace(microsecond=0)

        nfb_section.append(data)
    
    df = pd.DataFrame(nfb_section).astype({
        'RoadClass': 'int64',
        'SectionLength': 'float64',
        'SpeedLimit': 'int64'
    })
    df.index += 1
    return df

def getNFBVdStatic(filename: str) -> pd.DataFrame:
    """ 取得 VD 靜態資料 (資料來源: 高公局) """
    _, root = xmlParser(filename)
    vdStatic = []
    vds = root[-1]
    for vd in vds:
        data = {}
        for col in vd:
            if (col.tag == 'VDID'):
                vdid = col.text.replace('--', '-')
                data[col.tag] = vdid
                if (re.search(r"[A-Z]*-(\w*)-([A-Z])-([(\d+(\.\d+)?)]*)-(\w)", vdid).group(4) == 'M') or \
                    (re.search(r"[A-Z]*-(\w*)-([A-Z])-([(\d+(\.\d+)?)]*)-(\w)", vdid).group(4) == 'N'):
                    data['Mainlane'] = 1
                else:
                    data['Mainlane'] = 0
            elif (col.tag == 'LocationMile'):
                data[col.tag] = mileageTransform(col.text)
            elif (col.tag == 'DetectionLinks'):
                for subcol in col[0]:
                    data[subcol.tag] = subcol.text
            elif (col.tag == 'RoadSection'):
                pass
            else:
                data[col.tag] = col.text
            data['CreateTime'] = datetime.now().replace(microsecond=0)
        
        vdStatic.append(data)


    vdStatic = pd.DataFrame(vdStatic).astype({
        'BiDirectional': 'int64',
        'LaneNum': 'int64',
        'ActualLaneNum': 'int64',
        'VDType': 'int64',
        'LocationType': 'int64',
        'DetectionType': 'int64',
        'PositionLon': 'float64',
        'PositionLat': 'float64',
        'RoadClass': 'int64'
    })
    
    return vdStatic

def getNFBVdDynamic(filename: str) -> tuple:
    """ 取得 VD 動態資料 (資料來源: 高公局) """
    vdDymc = []
    vdDymcDtl = []
    _, root = xmlParser(filename)
    vdLives = root[-1]
    for vdLive in vdLives:
        data = {}
        for col in vdLive:
            if (col.tag == 'VDID'):
                vdid = col.text.replace('--', '-')
                data[col.tag] = vdid
            elif (col.tag == 'LinkFlows'):
                for subcol in col[0]:
                    if (subcol.tag == 'LinkID'):
                        data[subcol.tag] = subcol.text
                    else:
                        for lane in subcol:
                            dtlData = {}
                            for laneDtl in lane:
                                dtlData['VDID'] = vdid
                                if (laneDtl.tag == 'Vehicles'):
                                    dtlData['Volume'] = 0
                                    for veh in laneDtl:
                                        if (veh[0].text == 'S'):
                                            dtlData['SmallCarVolume'] = int(veh[1].text)
                                            if (int(veh[1].text) != -99):
                                                dtlData['Volume'] += int(veh[1].text)
                                            else:
                                                dtlData['Volume'] = -99
                                        elif (veh[0].text == 'L'):
                                            dtlData['LargeCarVolume'] = int(veh[1].text)
                                            if (int(veh[1].text) != -99):
                                                dtlData['Volume'] += int(veh[1].text)
                                            else:
                                                dtlData['Volume'] = -99
                                        elif (veh[0].text == 'T'):
                                            dtlData['TruckCarVolume'] = int(veh[1].text)
                                            if (int(veh[1].text) != -99):
                                                dtlData['Volume'] += int(veh[1].text)
                                            else:
                                                dtlData['Volume'] = -99
                                else:
                                    dtlData[laneDtl.tag] = laneDtl.text
                            vdDymcDtl.append(dtlData)
            elif (col.tag == 'DataCollectTime'):
                data[col.tag] = datetime.strptime(col.text, '%Y-%m-%dT%H:%M:%S+08:00')
                data['DataCollectTimeStamp'] = int(datetime.strptime(col.text, '%Y-%m-%dT%H:%M:%S+08:00').timestamp())
                data['CreateTime'] = datetime.now().replace(microsecond=0)
            else:
                data[col.tag] = col.text

        vdDymc.append(data)

    vdDymc = pd.DataFrame(vdDymc).astype({
        'Status': 'int64'
    })

    vdDymcDtl = pd.DataFrame(vdDymcDtl).astype({
        'LaneID': 'int64',
        'LaneType': 'int64',
        'Speed': 'int64',
        'Occupancy': 'int64'
    })
    vdDymcDtl['DataCollectTime'] = [vdDymc['DataCollectTime'].iloc[0] for _ in range(vdDymcDtl.shape[0])]
    vdDymcDtl['DataCollectTimeStamp'] = vdDymcDtl['DataCollectTime'].apply(lambda x: int(x.tz_localize('Asia/Taipei').timestamp()))
    vdDymcDtl['CreateTime'] = [vdDymc['CreateTime'].iloc[0] for _ in range(vdDymcDtl.shape[0])]

    return vdDymc, vdDymcDtl

def getTWRoadData(filename: str) -> tuple:
    """ 取得全臺道路基本資料 """
    roadInfo = pd.read_csv(filename).astype({'UpdateDate': 'datetime64[ns]'})
    roadInfo = roadInfo.rename(columns={'UpdateDate': 'UpdateTime'})
    roadInfo['CreateTime'] = [datetime.now().replace(microsecond=0) for _ in range(roadInfo.shape[0])]

    roadClassDf = roadInfo.groupby(by='RoadClass').agg({
        'RdCName': 'max',
        'CreateTime': 'max'
    }).reset_index()
    roadInfo = roadInfo[['RoadID', 'RoadClass', 'RoadName', 'UpdateTime', 'CreateTime']]
    roadInfo = roadInfo.groupby(by=['RoadID', 'RoadName']).agg({
        'RoadClass': 'max',
        'UpdateTime': 'max',
        'CreateTime': 'max'
    }).reset_index()
    roadInfo.index += 1
    roadClassDf.index += 1
    return roadInfo, roadClassDf

def getTHBVdStatic(filename: str) -> pd.DataFrame:
    """ 取得 VD 靜態資料 (資料來源: 公路局) """
    _, root = xmlParser(filename)
    vdStatic = []
    vds = root[-2]
    for vd in vds:
        if (vd[2].text == '1'):
            data_0, data_1 = {}, {}
            for col in vd:
                if (col.tag == 'DetectionLinks'):
                    for subcol in col[0]:
                        data_0[subcol.tag] = subcol.text
                    for subcol in col[1]:
                        data_1[subcol.tag] = subcol.text
                else:
                    data_0[col.tag] = col.text
                    data_1[col.tag] = col.text

                data_0['CreateTime'] = datetime.now().replace(microsecond=0)
                data_1['CreateTime'] = datetime.now().replace(microsecond=0)

            vdStatic.append(data_0)
            vdStatic.append(data_1)
        
        else:
            data = {}
            for col in vd:
                if (col.tag == 'DetectionLinks'):
                    for subcol in col[0]:
                        data[subcol.tag] = subcol.text
                else:
                    data[col.tag] = col.text
                data['CreateTime'] = datetime.now().replace(microsecond=0)
            vdStatic.append(data)

    vdStatic = pd.DataFrame(vdStatic).astype({
        'BiDirectional': 'int64',
        'LaneNum': 'int64',
        'ActualLaneNum': 'int64',
        'VDType': 'int64',
        'LocationType': 'int64',
        'DetectionType': 'int64',
        'PositionLon': 'float64',
        'PositionLat': 'float64',
        'RoadClass': 'int64',
        'CreateTime': 'datetime64[ns]'
    })
    vdStatic['LocationMile'] = vdStatic['VDID'].apply(lambda x: float(re.search(r"[A-Z]*-\d*-\w*-(\d*)-\d*", x).group(1)))
    vdStatic['Abnormal'] = [0 for _ in range(vdStatic.shape[0])]
    return vdStatic


if __name__ == '__main__':
    sql = " SELECT * FROM vd_static_2023 "
    vdStatic = pd.read_sql(sql, con=engine)
    vdStaticIDs = {k: v for k, v in zip(vdStatic['VDID'], vdStatic['id'])}

    dtList = list(map(lambda x: str(x), list(pd.date_range('2024-01-01', '2024-04-15'))))
    for dt in dtList:
        date = dt[:10]
        dtEnd = str((datetime.strptime(dt, '%Y-%m-%d %H:%M:%S') + timedelta(days=1)).replace(microsecond=0))
        
        with engine.connect() as conn:
            sql = text(f"""
            CREATE TABLE vd_dynamic_detail_{date.replace('-','')} (
                id INT PRIMARY KEY AUTO_INCREMENT COMMENT '流水號',
                VdStaticID INT COMMENT 'ref. vd_static.id',
                FOREIGN KEY (VdStaticID) REFERENCES vd_static_2023(id),
                LaneID INT COMMENT '車道代碼',
                LaneType INT COMMENT '車道種類 = [1: 一般車道, 2: 快車道, 3: 慢車道, 4: 機車道, 5: 高承載車道, 6: 公車專用道, 7: 轉向車道, 8: 路肩, 9: 輔助車道, 10: 調撥車道, 11: 其他]',
                Speed INT COMMENT '平均速率偵測值, [-99:資料異常, -1:道路封閉]',
                Occupancy INT COMMENT '佔有率偵測值',
                Volume INT COMMENT '流量偵測值, -99=資料異常',
                SmallCarVolume INT COMMENT '小型車流量偵測值',
                LargeCarVolume INT COMMENT '大型車流量偵測值',
                TruckCarVolume INT COMMENT '聯結車流量偵測值',
                DataCollectTime DATETIME COMMENT '資料蒐集時間',
                CreateTime DATETIME COMMENT '建立時間'
            ) COMMENT '{date} VD 動態資料';
            """)
            conn.execute(sql)
            conn.commit()
            conn.close()
            logging.debug(f"Table `vd_dynamic_detail_{date.replace('-','')}` has been created.")
                    
        
        for hour in range(24):
            for minute in range(60):
                rtn = vd.download(date.replace('-',''), hour, minute, directory='./')
                if rtn:
                    logging.debug(rtn)
                    continue
                else:
                    nfbVdDymc, nfbVdDymcDtl = getNFBVdDynamic(f"./VDLive_{date.replace('-','')}/VDLive_{hour:02d}{minute:02d}.xml.gz")
                
                nfbVdDymc = nfbVdDymc.loc[(nfbVdDymc['VDID'].str.contains('N5-')) & (nfbVdDymc['VDID'].isin(vdStaticIDs.keys()))].reset_index(drop=True)
                nfbVdDymcDtl = nfbVdDymcDtl.loc[(nfbVdDymcDtl['VDID'].str.contains('N5-')) & (nfbVdDymcDtl['VDID'].isin(vdStaticIDs.keys()))].reset_index(drop=True)

                # 3NF Process
                nfbVdDymc['VDID'] = nfbVdDymc['VDID'].apply(lambda x: vdStaticIDs[x])
                nfbVdDymc.rename(columns={'VDID': 'VdStaticID'}, inplace=True)
                nfbVdDymcDtl['VDID'] = nfbVdDymcDtl['VDID'].apply(lambda x: vdStaticIDs[x])
                nfbVdDymcDtl.rename(columns={'VDID': 'VdStaticID'}, inplace=True)
                                    
                # Transfer nan to None, or it will raise ProgrammingError in pymysql.err
                nfbVdDymc = nfbVdDymc.where(nfbVdDymc.notnull(), None)
                nfbVdDymcDtl = nfbVdDymcDtl.where(nfbVdDymcDtl.notnull(), None)
                
                # Get the info of table
                metadata = MetaData()
                metadata.reflect(bind=engine)
                
                logging.debug(f"Writing to db: {date.replace('-','')}_{hour:02d}:{minute:02d}")
                with engine.connect() as conn:
                    conn.execute(
                        insert(metadata.tables[f"vd_dynamic_detail_{date.replace('-','')}"]),
                        nfbVdDymcDtl.to_dict(orient='records')
                    )
                    conn.commit()
                    conn.close()
        shutil.rmtree(f"VDLive_{date.replace('-','')}")
        logging.debug(f"Finished `vd_dynamic_detail_{date.replace('-','')}`.")

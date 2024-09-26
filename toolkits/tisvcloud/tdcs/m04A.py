""" Created on May, 2022
    Coder: CH
"""

import os
import shutil
import requests
import tarfile
import pandas as pd


__version__ = "0.1.0"
__domain_url__ = "https://tisvcloud.freeway.gov.tw/history/TDCS/M04A"

def _check_empty(path):
    size = os.path.getsize(path)
    if size < 1024*1024:
        KB = round(size/1024, 2)
        if KB < 1:
            return True
        else:
            return False

def _trans_format(hour, minute):
    hour = int(hour)
    minute = int(minute)
    hour = f"{hour:02d}"
    minute = f"{minute:02d}"
    return hour, minute

def download(date, hour, minute):
    hour, minute = _trans_format(hour, minute)
    download_url = f"{__domain_url__:s}/{date}/{hour}/TDCS_M04A_{date}_{hour}{minute}00.csv"
    r = requests.get(download_url)
    with open(f"TDCS_M04A_{date}_{hour}{minute}00.csv", "wb") as f:
        f.write(r.content)
        f.close()
    filepath = f"TDCS_M04A_{date}_{hour}{minute}00.csv"
    # check whether the file is empty:
    if _check_empty(filepath):
        os.remove(filepath)
        
        # 年代久遠的資料會壓縮成tar.gz檔
        r = requests.get(f"{__domain_url__}/M04A_{date}.tar.gz")
        tarfile_path = f"M04A_{date}.tar.gz"
        with open(tarfile_path, "wb") as f:
            f.write(r.content)
            f.close()
        
        # 解壓縮後把壓縮檔移除
        with tarfile.open(f"M04A_{date}.tar.gz") as f:
            f.extractall()
        os.remove(f"M04A_{date}.tar.gz")
        
        # 取得相關資料
        df = readfile(f"M04A/{date}/{hour}/TDCS_M04A_{date}_{hour}{minute}00.csv")
    return df

def read_original_file(filepath):
    df = pd.read_csv(filepath)
    return df
    
def readfile(filepath):
    columns = ["Detected_Time", "Gantry_ID", "Direction", "VehType", "TravTime", "Volume"]
    df = pd.read_csv(filepath, names=columns)
    df = df.astype({"VehType": "string"})
    return df

def get_volume(gantry_ID, date, hour, minute):
    hour, minute = _trans_format(hour, minute)
    download(date, hour, minute)
    filepath = f"TDCS_M04A_{date}_{hour}{minute}00.csv"
    
    # check whether the file is empty:
    if _check_empty(filepath):
        os.remove(filepath)
        
        # 年代久遠的資料會壓縮成tar.gz檔
        r = requests.get(f"{__domain_url__}/M04A_{date}.tar.gz")
        tarfile_path = f"M04A_{date}.tar.gz"
        with open(tarfile_path, "wb") as f:
            f.write(r.content)
            f.close()
        
        # 解壓縮後把壓縮檔移除
        with tarfile.open(f"M04A_{date}.tar.gz") as f:
            f.extractall()
        os.remove(f"M04A_{date}.tar.gz")
        
        # 取得相關資料
        df = readfile(f"M04A/{date}/{hour}/TDCS_M04A_{date}_{hour}{minute}00.csv")
        filted_df = df.loc[df["Gantry_ID"] == gantry_ID].reset_index(drop=True)
        eachVehType_vol = {}
        for i in range(filted_df.shape[0]):
            eachVehType_vol[filted_df["VehType"].iloc[i]] = filted_df["Volume"].iloc[i]
            
        # 將資料夾刪除
        shutil.rmtree("M04A/")
        
    else:
        df = readfile(filepath)
        filted_df = df.loc[df["Gantry_ID"] == gantry_ID].reset_index(drop=True)
        eachVehType_vol = {}
        for i in range(filted_df.shape[0]):
            eachVehType_vol[filted_df["VehType"].iloc[i]] = filted_df["Volume"].iloc[i]
        
        os.remove(f"TDCS_M04A_{date}_{hour}{minute}00.csv")
    
    return eachVehType_vol

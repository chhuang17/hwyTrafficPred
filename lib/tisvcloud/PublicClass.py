# This is a public module for all other tisvcloud-related modules to use.
# Created on May 17, 2023
# Author: Chien-Hao Huang


__version__ = "0.1.0"


import os
import shutil
import requests
import tarfile
import pandas as pd
import time
from urllib.parse import urljoin


class TisvDownload:
    def __init__(self, date, hour, minute):
        self.domain_url = "https://tisvcloud.freeway.gov.tw/history/"
        self.date = date
        self.hour = hour
        self.minute = minute        
    
    def empty(self, path):
        size = os.path.getsize(path)
        if size < 1024*1024:
            KB = round(size/1024, 2)
            if KB < 1:
                return True
            else:
                return False

    def trans_format(self):
        if type(self.date) != str:
            self.date = int(self.date)
            self.date = str(self.date)
        if type(self.hour) != str:
            self.hour = int(self.hour)
            self.hour = f"{self.hour:02d}"
        if type(self.minute) != str:
            self.minute = int(self.minute)
            self.minute = f"{self.minute:02d}"
    
    
class TDCSData(TisvDownload):
    def __init__(self, date, hour, minute, dataname):
        super().__init__(date, hour, minute)
        self.dataname = dataname
        self.trans_format()
        self.download_url = urljoin(self.domain_url, f"TDCS/{self.dataname}/")
        self.folder = f"{self.dataname}_{self.date}/"
        self.filename = os.path.join(self.folder,
                                     f"TDCS_{self.dataname}_{self.date}_{self.hour}{self.minute}00.csv")
    
    def empty(self):
        return super().empty(self.filename)
        
    def trans_format(self):
        super().trans_format()
    
    def download(self):
        if os.path.isdir(self.folder):
            pass
        else:
            os.makedirs(self.folder)
        
        download_url = f"{self.download_url:s}/{self.date}/{self.hour}/TDCS_{self.dataname}_{self.date}_{self.hour}{self.minute}00.csv"
        r = requests.get(download_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Connection': 'close'
        })
        with open(self.filename, "wb") as f:
            f.write(r.content)
            f.close()
            r.close()
            time.sleep(1)
        
        # check whether the file is empty:
        if self.empty():
            os.remove(self.filename)
        
            # 年代久遠的資料會壓縮成tar.gz檔
            r = requests.get(f"{self.download_url}/{self.dataname}_{self.date}.tar.gz")
            tarfile_path = f"{self.folder}{self.dataname}_{self.date}.tar.gz"
            with open(tarfile_path, "wb") as f:
                f.write(r.content)
                f.close()

            # 解壓縮後把壓縮檔移除
            with tarfile.open(f"{self.folder}{self.dataname}_{self.date}.tar.gz") as f:
                f.extractall(path=self.folder)
            os.remove(f"{self.folder}{self.dataname}_{self.date}.tar.gz")

            self.filename = f"{self.folder}{self.dataname}/{self.date}/{self.hour}/TDCS_{self.dataname}_{self.date}_{self.hour}{self.minute}00.csv"

    def delete(self):
        shutil.rmtree(self.folder)
            
            
class MOTCLiveData(TisvDownload):
    def __init__(self, date, hour, minute, dataname, directory='./'):
        super().__init__(date, hour, minute)
        self.dataname = dataname
        self.trans_format()
        self.download_url = urljoin(self.domain_url, f"motc20/{dataname}/")
        self.folder = os.path.join(directory, f"{self.dataname}Live_{self.date}/")
        self.filename = os.path.join(self.folder, f"{self.dataname}Live_{self.hour}{self.minute}.xml.gz")
        
    def empty(self):
        return super().empty(self.filename)
        
    def trans_format(self):
        super().trans_format()
    
    def download(self):
        if os.path.isdir(self.folder):
            pass
        else:
            os.makedirs(self.folder)
        
        download_url = f"{self.download_url:s}/{self.date}/{self.dataname}Live_{self.hour}{self.minute}.xml.gz"
        r = requests.get(download_url)
        
        with open(self.filename, "wb") as f:
            f.write(r.content)
            f.close()
        
    def delete(self):
        os.remove(self.filename)
        # shutil.rmtree(self.folder)


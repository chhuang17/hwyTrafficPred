from tisvcloud.PublicClass import MOTCLiveData
import xml.etree.ElementTree as ET
import gzip
import numpy as np


__version__ = "0.1.0"

# Hierachy 0
# CMSLives:          root[-1]

# Hierachy 1
# CMSLive:           root[-1][i]                  (=CMSLives[i])

# Hierachy 2
# CMSID:             root[-1][i][0].text          (=CMSLives[i][0].text)
# MessageStatus:     root[-1][i][1].text          (=CMSLives[i][1].text)
# Messages:          root[-1][i][2]               (=CMSLives[i][2])
# Status:            root[-1][i][3].text          (=CMSLives[i][3].text)
# DataCollectTime:   root[-1][i][-1].text         (=CMSLives[i][-1].text)

# Hierachy 3
# Message:           root[-1][i][2][0]            (=CMSLives[i][2][0])

# Hierachy 4
# Text:              root[-1][i][2][0][0].text    (=CMSLives[i][2][0][0].text)



def download(CMSID, date, hour, minute):
    cmslive = MOTCLiveData(date, hour, minute, dataname="CMS")
    cmslive.download()
    if cmslive.empty():
        cmslive.delete()
        return f"Cannot find CMSLiveData: {cmslive.date}/CMSLive_{cmslive.hour}{cmslive.minute}.xml"

def message(CMSID, date, hour, minute):
    cmslive = MOTCLiveData(date, hour, minute, dataname="CMS")
    cmslive.download()
    if cmslive.empty():
        cmslive.delete()
        return f"Cannot find CMSLiveData: {cmslive.date}/CMSLive_{cmslive.hour}{cmslive.minute}.xml"
    
    with gzip.open(cmslive.filename, "r") as xmlfile:
        try:
            tree = ET.parse(xmlfile)
            root = tree.getroot()
            CMSLives = root[-1]
            
        except ET.ParseError:
            return f"{cmslive.filename} is Empty!"
            
        else:
            for i in range(len(CMSLives)):
                if CMSID == CMSLives[i][0].text:
                    if CMSLives[i][3].text == "0":
                        if CMSLives[i][1].text == "0":
                            xmlfile.close()
                            cmslive.delete()
                            return f"{CMSID} 目前無資料顯示!"
                        elif CMSLives[i][1].text == "1":
                            xmlfile.close()
                            cmslive.delete()
                            return CMSLives[i][2][0][0].text
                    
                    
                    elif CMSLives[i][3].text == "1":
                        xmlfile.close()
                        cmslive.delete()
                        return f"{CMSID} 通訊異常!"
                    
                    elif CMSLives[i][3].text == "2":
                        xmlfile.close()
                        cmslive.delete()
                        return f"{CMSID} 停用或施工中!"
                    
                    elif CMSLives[i][3].text == "3":
                        xmlfile.close()
                        cmslive.delete()
                        return f"{CMSID} 設備故障!"
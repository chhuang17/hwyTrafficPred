from tisvcloud.PublicClass import TDCSData
import pandas as pd
import numpy as np


__version__ = "0.1.0"



def download(dataname, date, hour, minute):
    etcData = TDCSData(date, hour, minute, dataname=dataname)
    etcData.download()
    if etcData.empty():
        return f"Cannot find etcData: TDCS_{etcData.dataname}_{etcData.date}_{etcData.hour}{etcData.minute}00.csv"

def volume(GantryID, date, hour, minute):
    etcData = TDCSData(date, hour, minute, dataname="M03A")
    etcData.download()
    if etcData.empty():
        return f"Cannot find etcData: TDCS_{etcData.dataname}_{etcData.date}_{etcData.hour}{etcData.minute}00.csv"
    
    df = pd.read_csv(
        etcData.filename,
        names=["TimeInterval", "GantryID", "Direction", "VehType", "Traffic"]
    )
    df = df.astype({"VehType": "string"})
    filted_df = df.loc[df["GantryID"] == GantryID].reset_index(drop=True)
    eachVehType_vol = {}
    for i in range(filted_df.shape[0]):
        eachVehType_vol[filted_df["VehType"].iloc[i]] = int(filted_df["Traffic"].iloc[i])   
    etcData.delete()
    return eachVehType_vol
    
def medianTravTime(GantryFrom, GantryTo, date, hour, minute):
    etcData = TDCSData(date, hour, minute, dataname="M04A")
    etcData.download()
    if etcData.empty():
        return f"Cannot find etcData: TDCS_{etcData.dataname}_{etcData.date}_{etcData.hour}{etcData.minute}00.csv"
    
    df = pd.read_csv(
        etcData.filename,
        names=["TimeInterval", "GantryFrom", "GantryTo", "VehType", "TravelTime", "Traffic"]
    )
    df = df.astype({"VehType": "string"})
    filted_df = df.loc[(df["GantryFrom"]==GantryFrom) & (df["GantryTo"]==GantryTo)].reset_index(drop=True)
    eachVehType_travtime = {}
    for i in range(filted_df.shape[0]):
        eachVehType_travtime[filted_df["VehType"].iloc[i]] = int(filted_df["TravelTime"].iloc[i])        
    return eachVehType_travtime

def medianSpeed(GantryFrom, GantryTo, date, hour, minute):
    etcData = TDCSData(date, hour, minute, dataname="M05A")
    etcData.download()
    if etcData.empty():
        return f"Cannot find etcData: TDCS_{etcData.dataname}_{etcData.date}_{etcData.hour}{etcData.minute}00.csv"
    
    df = pd.read_csv(
        etcData.filename,
        names=["TimeInterval", "GantryFrom", "GantryTo", "VehType", "SpaceMeanSpeed", "Traffic"]
    )
    df = df.astype({"VehType": "string"})
    filted_df = df.loc[(df["GantryFrom"]==GantryFrom) & (df["GantryTo"]==GantryTo)].reset_index(drop=True)
    eachVehType_speed = {}
    for i in range(filted_df.shape[0]):
        eachVehType_speed[filted_df["VehType"].iloc[i]] = int(filted_df["SpaceMeanSpeed"].iloc[i])
    return eachVehType_speed
    
def tripData(date, hour, minute):
    etcData = TDCSData(date, hour, minute, dataname="M06A")
    etcData.download()
    if etcData.empty():
        return f"Cannot find etcData: TDCS_{etcData.dataname}_{etcData.date}_{etcData.hour}{etcData.minute}00.csv"
    
    df = pd.read_csv(
        etcData.filename,
        names=["VehType", "DetectTime_O", "GantryID_O", "DetectTime_D", "GantryID_D", "TripLength", "TripEnd"]
    )
    df = df.astype({"VehType": "string"})
    return df
    
def tripLength(GantryID, date, hour, minute):
    etcData = TDCSData(date, hour, minute, dataname="M07A")
    etcData.download()
    if etcData.empty():
        return f"Cannot find etcData: TDCS_{etcData.dataname}_{etcData.date}_{etcData.hour}{etcData.minute}00.csv"
    
    df = pd.read_csv(
        etcData.filename,
        names=["TimeInterval", "GantryFrom", "VehType", "TripLength(km)", "Traffic"]
    )
    df = df.astype({"VehType": "string"})
    filted_df = df.loc[(df["GantryFrom"]==GantryID)].reset_index(drop=True)
    eachVehType_tripLength = {}
    for i in range(filted_df.shape[0]):
        eachVehType_tripLength[filted_df["VehType"].iloc[i]] = round(filted_df["TripLength(km)"].iloc[i], 1)
    return eachVehType_tripLength
    
def odPair(GantryFrom, GantryTo, date, hour, minute):
    etcData = TDCSData(date, hour, minute, dataname="M08A")
    etcData.download()
    if etcData.empty():
        return f"Cannot find etcData: TDCS_{etcData.dataname}_{etcData.date}_{etcData.hour}{etcData.minute}00.csv"
    
    df = pd.read_csv(
        etcData.filename,
        names=["TimeInterval", "GantryFrom", "GantryTo", "VehType", "Traffic"]
    )
    df = df.astype({"VehType": "string"})
    filted_df = df.loc[(df["GantryFrom"]==GantryFrom) & (df["GantryTo"]==GantryTo)].reset_index(drop=True)
    eachVehType_odPair = {}
    for i in range(filted_df.shape[0]):
        eachVehType_odPair[filted_df["VehType"].iloc[i]] = int(filted_df["Traffic"].iloc[i])
    return eachVehType_odPair

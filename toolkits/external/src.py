from datetime import datetime
import pandas as pd
import numpy as np


def _dateFormat(x):
    date = datetime.strptime(str(x), '%Y%m%d')
    date = datetime.strftime(date, '%Y-%m-%d')
    return date

def _weekdayFmt(x):
    if x == '一':
        return 1
    elif x == '二':
        return 2
    elif x == '三':
        return 3
    elif x == '四':
        return 4
    elif x == '五':
        return 5
    elif x == '六':
        return 6
    elif x == '日':
        return 7

def _isConsecutive(idList):
    """ 判斷傳入的陣列是否為連續數字組成 """
    i = 0
    while (i < len(idList)-1):
        if (idList[i]+1 != idList[i+1]):
            return False
        i += 1
    return True

def _addHolidays(notToWork, calendar):
    """ 國定假日一定會備註, 然後一般來說連假就是看國定假日的前二後二有沒有放假 \n
        所以關注在備註欄不為空的部分, 取前二後二會得到一個小的 dataframe, 並用 id 那一個欄位去判斷 \n
        如果 id 是連續的, 表示那幾天就是連假, 並在 calendar 的 isHoliday 那一欄相對應的位置填上 1 """
    i = 0
    while i < notToWork.shape[0]:
        if pd.notna(notToWork['Note'].iloc[i]):
            lower = max(0, i-2)
            upper = min(i+3, notToWork.shape[0]-1)
            if (lower == 0):
                idList = list(notToWork.iloc[lower:i+2,:]['id'])
            else:
                idList = list(notToWork.iloc[lower:i+1,:]['id'])

            if _isConsecutive(idList):
                calendar.loc[list(map(lambda x: x-1, idList)), 'isHoliday'] = np.int32(1)
            
            if (upper == notToWork.shape[0]-1):
                idList = list(notToWork.iloc[i-1:upper,:]['id'])
            else:
                idList = list(notToWork.iloc[i:upper,:]['id'])

            if _isConsecutive(idList):
                calendar.loc[list(map(lambda x: x-1, idList)), 'isHoliday'] = np.int32(1)
        
        i += 1

    return calendar

def Calendar(year: int = 2023) -> pd.DataFrame:
    calendar = pd.read_csv(f"../hwyTrafficPred/toolkits/external/calendar/calendar_{year}.csv", encoding='Big5')
    calendar.insert(0, 'id', list(map(lambda x: x+1, list(calendar.index))))
    calendar = calendar.rename(columns={'西元日期': 'Date', '星期': 'Weekday', '是否放假': 'offWork', '備註': 'Note'})

    calendar['Date'] = calendar['Date'].apply(lambda x: _dateFormat(x))
    calendar['Weekday'] = calendar['Weekday'].apply(lambda x: _weekdayFmt(x))
    calendar['offWork'] = calendar['offWork'].apply(lambda x: x-1 if x > 0 else x)

    # 新增欄位判斷是否為連續假期
    calendar['isHoliday'] = pd.Series(np.zeros(calendar.shape[0], dtype=np.int32))

    notToWork = calendar.loc[calendar['offWork']==1].reset_index(drop=True)
    calendar = _addHolidays(notToWork, calendar)

    calendar.drop(columns=['id'], inplace=True)
    calendar = calendar.reindex(columns=['Date', 'Weekday', 'offWork', 'isHoliday', 'Note'])
    calendar = calendar.astype({'Weekday': 'int', 'offWork': 'int', 'isHoliday': 'int'})

    return calendar
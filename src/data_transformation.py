import numpy as np
import pandas as pd
from datetime import datetime


def read_file(path):
    file = pd.read_csv(path)
    return file


def remove_outlier(file):
    thrsh = 300
    toBeDropped = []
    for column in file.columns:

        for index, entry in enumerate(file[column]):
            try:
                entry += 0
            except TypeError:
                break
            # cant compare at beginning and end
            if index in [0, 1] or index == len(column)-1:
                continue
            if entry > thrsh * file.at[index-1, column] and entry > thrsh * file.at[index+1, column]:
                if index not in toBeDropped:
                    toBeDropped.append(index)
                continue
            if entry * thrsh < file.at[index-1, column] and entry * thrsh < file.at[index+1, column]:
                if index not in toBeDropped:
                    toBeDropped.append([index])
                continue
    for index in toBeDropped:
        file = file.drop(axis=0, index=index)
    file = file.reset_index(drop=True)
    return file


def process_file(file):
    file['snow_1h'] = file['snow_1h'] / 0.51
    #############################################
    file['rain_1h'] = file['rain_1h']/31.75
    #############################################
    file['clouds_all'] = file['clouds_all']/100
    #############################################
    file['temp'] = file['temp'] /310.07
    #############################################
    index = 0
    # make 'holiday' a binary
    x = 'holiday'
    for i in file[x]:
        if i == 'None':
            file.at[index, x] = 0
            index += 1
        else:
            file.at[index, x] = 1
            index += 1
    #############################################
    # make 'weather_main' normalized
    index = 0
    weather_main = {'Clouds':0, 'Clear':1, 'Rain':2, 'Drizzle':3, 'Mist':4, 'Haze':5, 'Fog':6,
                    'Thunderstorm':7, 'Snow':8, 'Squall':9, 'Smoke':10}
    x = 'weather_main'
    for i in file[x]:
        file.at[index, x] = weather_main[i]/10
        index += 1
    #############################################
    # make 'weather_description' normalized
    x = 'weather_description'
    index = 0
    weather_description = {'scattered clouds': 0, 'broken clouds': 1, 'overcast clouds': 2, 'sky is clear': 3,
                           'few clouds': 4, 'light rain': 5, 'light intensity drizzle': 6, 'mist': 7, 'haze': 8,
                           'fog': 9, 'proximity shower rain': 10, 'drizzle': 11, 'moderate rain': 12,
                           'heavy intensity rain': 13,
                           'proximity thunderstorm': 14, 'thunderstorm with light rain': 15,
                           'proximity thunderstorm with rain': 16,
                           'heavy snow': 17, 'heavy intensity drizzle': 18, 'snow': 19,
                           'thunderstorm with heavy rain': 20,
                           'freezing rain': 21, 'shower snow': 22, 'light rain and snow': 23,
                           'light intensity shower rain': 24,
                           'SQUALLS': 25, 'thunderstorm with rain': 26, 'proximity thunderstorm with drizzle': 27,
                           'thunderstorm': 28,
                           'Sky is Clear': 29, 'very heavy rain': 30, 'thunderstorm with light drizzle': 31,
                           'light snow': 32,
                           'thunderstorm with drizzle': 33, 'smoke': 34, 'shower drizzle': 35, 'light shower snow': 36,
                           'sleet': 37}

    for i in file[x]:
        file.at[index, x] = weather_description[i] / 37
        index += 1
    #############################################
    file["date_time"] = file["date_time"].apply(lambda x: \
                                                   datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    file['hour'] = [d.hour for d in file['date_time']]
    file['day'] = [d.day for d in file['date_time']]
    file['month'] = [d.month for d in file['date_time']]
    file['year'] = [(d.year -2012)/6 for d in file['date_time']]
    file['timeX'] = np.sin(2.*np.pi*(file["hour"]/24))
    file['timeY'] = np.cos(2. * np.pi * (file["hour"] / 24))
    file['monthX'] = np.sin(2. * np.pi * (file["month"] / 12))
    file['monthY'] = np.cos(2. * np.pi * (file["month"] / 12))

    dayX = []
    dayY = []
    index = 0
    for day in file['day']:
        if file.at[index, 'month'] in [1, 3, 5, 7, 8, 10, 12]:
            dayX.append(np.sin(2. * np.pi * (day / 31)))
            dayY.append(np.cos(2. * np.pi * (day / 31)))
        elif file.at[index, 'month'] in [4, 6, 9, 11]:
            dayX.append(np.sin(2. * np.pi * (day / 30)))
            dayY.append(np.cos(2. * np.pi * (day / 30)))
        elif file.at[index, 'month'] in [2]:
            if file.at[index, 'year'] % 4 == 0:
                dayX.append(np.sin(2. * np.pi * (day / 29)))
                dayY.append(np.cos(2. * np.pi * (day / 29)))
            else:
                dayX.append(np.sin(2. * np.pi * (day / 28)))
                dayY.append(np.cos(2. * np.pi * (day / 28)))
    file['dayX'] = dayX
    file['dayY'] = dayY
    #############################################
    # cleanup :
    del file['date_time']
    del file['hour']
    del file['day']
    del file['month']
    temp = file.pop('traffic_volume')
    file['traffic_volume'] = temp




    return file

def find_types(file):
    types = []
    x = 'clouds_all'
    big = 0.01
    index = 0
    indexa = 0
    for i in file[x]:
        indexa += 1
        if i > big:
            big = i
            index = indexa
        else:
            continue
    # for i in file[x]:
    #     if i in types:
    #         continue
    #     else:
    #         types.append(i)


    print(big, index)

def write_csv(file):
    file.to_csv(path_or_buf="newFile.csv")


if __name__ == "__main__":
    path = "Metro_Interstate_Traffic_Volume.csv"
    file = read_file(path)
    file = remove_outlier(file)
    processedFile = process_file(file)
    write_csv(processedFile)
    #find_types(file)

# Necessary Imports

from datetime import datetime as dt
from datetime import date,timedelta
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import os, sys, csv
# pd.set_option('display.max_columns', 50000)
# pd.set_option('display.width', 50000)

#Retreive all magnetograms that exists in our directory and save it as a csv
def check_files():
    source = '/data/hmi_jpgs_512/'
    with open('data_labels/totalfiles_jpg_512.csv','w',newline='',encoding='utf-8-sig') as f:
        w = csv.writer(f)
        for path, subdirs, files in os.walk(source):
            for name in files:
                w.writerow([os.path.join(path, name)])



def create_labels(df, hrs):

    """
    This function expects the "goes_flare_integrated.csv" as pandas dataframe and creates labels for each magnetogram instance
    where the prediction window is 24 hours preset using two timestamps pws and pwe. If we want to change the prediction window to some other period of time:
    then make sure to adjust the default pws= "2010-12-01 00:00:00" (this time indicates the start year and month when the jp2s are available from helioviewer)
    and default pwe= "2010-12-01 23:59:59" which verifies the total duration of 24 hrs. Adjusting pwe will be sufficient to adjust the prediction window.
    For example: set pwe= "2010-12-01 11:59:59" makes the duration to 12 hours.
    The variable "hrs" is used as a sampling window, i.e. if we want to label space magnetograms available every hours,
    then use hrs=1 and for bidaily (two instances per day) use hrs=12 . The variable stop is used to terminate the loop, which indicates end of 2018.


    This function will return a dataframe with 7 columns:
    'label': timestamp of the magnetogram
    'goes_class': Max flare Goes class observed with in the prediction window of given magnetogram timestamp
    'fl_lon': heliographic longitude of the max flare
    'fl_lat': heliographic latitude of the max flare
    'rest_fl': contains all other flare events (if any) besides max flare event as a list.
    'rest_lon' : heliographic longitudes of other flares in the same order of rest_fl as a list.
    'rest_lat':  heliographic latitudes of other flares in the same order of rest_fl as a list.


    Note: label and goes_class are the only important columns to train the model. rest of the columns are created for post analysis if and whenever needed.
    """

    #Prediction window Start
    pws = pd.to_datetime('2010-12-01 00:00:00',format='%Y-%m-%d %H:%M:%S')

    #Prediction Window Stop
    pwe = pd.to_datetime('2010-12-01 23:59:59',format='%Y-%m-%d %H:%M:%S')

    #Data available till 2018-12-30
    stop = pd.to_datetime('2018-12-31 23:59:59',format='%Y-%m-%d %H:%M:%S')

    #Datetime 
    df['start'] = pd.to_datetime(df['start_time'], format='%Y-%m-%d %H:%M:%S')

    #New Empty DF
    emp = pd.DataFrame()

    #List to store intermediate results
    lis = []
    cols=['label', 'goes_class', 'fl_lon', 'fl_lat', 'rest_fl', 'rest_lon', 'rest_lat']

    #Loop to check max from midnight to midnight and noon to noon
    while True:
        rest_fl= []
        rest_lon= []
        rest_lat = []
        #Date with max intensity of flare with in the 24 hour window
        emp = df[ (df.start > pws) & (df.start <= pwe) ].sort_values('goes_class', ascending=False)
        emp.reset_index(inplace=True)
#         print(emp)
        if pd.Series(emp.goes_class).empty:
            ins = 'NF'
            lon = 'unk'
            lat = 'unk'
            rest_fl = []
            rest_lon = []
            rest_lat = []
        else:
            new_emp = emp.sort_values('goes_class', ascending=False).head(1).squeeze(axis=0)
            ins = new_emp.goes_class
            lon = new_emp.fl_lon
            lat = new_emp.fl_lat
            for i in range(len(emp)-1):
                rest_fl.append(emp.loc[i+1]['goes_class'])
                rest_lon.append(emp.loc[i+1]['fl_lon'])
                rest_lat.append(emp.loc[i+1]['fl_lat'])
        lis.append([pws, ins, lon, lat, rest_fl, rest_lon, rest_lat])
        pws = pws + pd.Timedelta(hours=hrs)
        pwe = pwe + pd.Timedelta(hours=hrs)
        if pwe >= stop:
            break

    df_result = pd.DataFrame(lis, columns=cols)
    print('Completed!')
    return df_result

#This function is used to convert the timestamps stored in column 'label' indicating the timestamp of the
#magnetogram instances to name of the actual magnetograms stored in our data directory.
def date_to_filename(df):
    cols=['label']
    for items in cols:

        df[items] = pd.to_datetime(df[items], format='%Y-%m-%d %H:%M:%S')

        #Renaming label(Date) to this format of file HMI.m2010.05.21_12.00.00 
        df[items] = df[items].dt.year.astype(str) + '/' \
            + df[items].dt.month.map("{:02}".format).astype(str) + '/'\
            + df[items].dt.day.map("{:02}".format).astype(str) + '/'+ 'HMI.m'+ df[items].dt.year.astype(str) + '.' \
            + df[items].dt.month.map("{:02}".format).astype(str) + '.'\
            + df[items].dt.day.map("{:02}".format).astype(str) + '_' \
            + df[items].dt.hour.map("{:02}".format).astype(str) + '.'\
            + df[items].dt.minute.map("{:02}".format).astype(str) + '.'\
            + df[items].dt.second.map("{:02}".format).astype(str) + '.jpg'
    
    return df

#Once the labels are created for each magnetogram instances specified by sampling window (hrs), we check whether the magnetogram instance for given 
#time stamp exists or not in our directory. Due to some reasons, the source of magnetogram jp2s :Helioveiwer.org, has missing instances.
#This function simply checks the created labels and avaliable files and remove unavailble ones from the dataframe.
def filter_files(df, hrs):
    data_df = df.copy()
    file = pd.read_csv('data_labels/totalfiles_jpg_512.csv', names=['filenames'])
    file_df = pd.DataFrame(file)
    file_df['filenames'] = file_df['filenames'].str.replace("/data/hmi_jpgs_512/","")
    data_df['label_mis'] = data_df.label.isin(file_df.filenames)
    data_df = data_df.applymap(lambda x: x if x else np.nan)
    data_df = data_df[data_df['label_mis'].notna()]
    data_df.fillna('', inplace=True)
    cols=['label', 'goes_class', 'fl_lon', 'fl_lat', 'rest_fl', 'rest_lon', 'rest_lat']
    data_df.to_csv(r'data_labels/NEW_full_dataset_cleaned_{hrs}_hours_with_loc_and_time_new.csv'.format(hrs=hrs), index=False, header=True, columns=cols)
    data_df.reset_index(inplace=True)
    return data_df


def binarize(df, modes):
    """
    This function binarizes the goes_class to 0 and 1 labels
    indicating noflare and flare created using goes_class: if GreaterThanOrEqualsToM then 1 else 0
    The variable modes is used to change the labels for different prediction mode. 
    i.e. if GreaterThanOrEqualsToC is Flare then pass modes='C' else pass modes='M'
    """
    #Empty space and nan values are filled with 0 in the goes_class column
    df.replace(np.nan, str(0), inplace=True)
    df.replace('', str(0), inplace=True)
    df['target'] = df['goes_class']

    #Replacing X and M class flare with 1 and rest with 0 in goes_class column
    if(modes=='M'):
        for i in range(len(df)):
            if (df.target[i][0] == 'X' or  df.target[i][0] == 'M'):
                df.target[i] = 1
            else:
                df.target[i] = 0
    else:
        for i in range(len(df)):
            if (df.target[i][0] == 'X' or  df.target[i][0] == 'M' or df.target[i][0] == 'C'):
                df.target[i] = 1
            else:
                df.target[i] = 0
    return df


#Creating time-segmented 4-Fold CV Dataset, where 9 months of data is used for training and rest 3 for validation
def create_CVDataset(df):
    cols=['label','target', 'goes_class', 'fl_lon', 'fl_lat', 'rest_fl', 'rest_lon', 'rest_lat']
    search_list = [['01', '02', '03'], ['04', '05', '06'], ['07', '08', '09'], ['10', '11', '12']]
    for i in range(4):
        search_for = search_list[i]
        mask = df['label'].apply(lambda row: row[21:23]).str.contains('|'.join(search_for))
        train = df[~mask]
        val = df[mask]
#         print(train['goes_class'].value_counts())
        print(val['target'].value_counts())
        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        train.to_csv(r'data_labels/Fold{i}_train.csv'.format(i=i+1), index=False, header=True, columns=cols)
        val.to_csv(r'data_labels/Fold{i}_val.csv'.format(i=i+1), index=False, header=True, columns=cols)

if __name__ == '__main__':
    #Load Original source for Goes Flare X-ray Flux 
    data = pd.read_csv (r'data_source/goes_flares_integrated.csv')


    #Convert to DataFrame
    dataframe = pd.DataFrame(data, columns= ['start_time','goes_class', 'fl_lon', 'fl_lat'])
    dataframe['fl_lon'] = dataframe['fl_lon'].replace(np.nan, 'UNK')
    dataframe['fl_lat'] = dataframe['fl_lat'].replace(np.nan, 'UNK')

    mode = 'M'

    #Calling functions in order
    # check_files()
    df_res = create_labels(dataframe,  1)
    df_res1 = date_to_filename(df_res.copy())
    df_res2 = filter_files(df_res1.copy(), 1)
    df_res3 = binarize(df_res2.copy(), mode)
    df_res4 = create_CVDataset(df_res3.copy())

    print(len(df_res3))
import array
from heapq import merge
from itertools import groupby
from locale import format_string
from pathlib import Path
from matplotlib import pyplot as plt, test
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
warnings.simplefilter("ignore", UserWarning)
#import the_module_that_warns


def reformat_dates(df): 
    """ Given a df, the function return the df with an added column named "just_date"
    that contain all the dates in the df in a uniformed format of dd/mm/yyyy """

    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df['just_date'] = pd.to_datetime(df['date']).dt.date
    newdate = dt.date(1998, 11, 8) #to prevent NaN problems, chosen a random date as a "null" date
    df['just_date'].replace({pd.NaT: newdate}, inplace=True)
    df['just_date'] = df['just_date'].apply(lambda d: d.strftime("%d/%m/%Y"))
    addcol = df.pop('just_date')
    df.insert(2, 'just_date', addcol)

def unite_same_day_visits_to_the_first(df):
    """ For a subject with more than one EEG recording in the same day, keep only the first visit """

    df = df.sort_values(by=['subject','date']) #sorting visits by subject, and then by date
    df = df.reset_index()
    df = df.drop_duplicates(subset=['subject', 'just_date'], keep='first') #removes visits of subjects with more than one visit a day, and leaves the first appearance
    df = df.reset_index()

def final_date_merge(df1, df2, t): 
    """ Merges df1 and df2 by subject and creates a final_date column for each visit, where t is the valid maximun difference in days.
    Considering only visits where the BNA visit and the clinical visit have at most t days between them, and uniting them 
    with one 'final_date'. Note that the function is dropping duplicates and keeping only the first appearance of a final date, 
    in order to prevent duplication of visits """

    df = df1.merge(df2, how = 'inner',on = ['subject']) #keeping in the df all matching pairs with the same subject
    df['final_date'] =[d1 if time_difference(d1,d2)<= t else "remove" for d1,d2 in zip(df['just_date_x'],df['just_date_y'])] 
    df = df[df['final_date']!= 'remove']
    df = df.reset_index()
    df = df.drop_duplicates(subset=['final_date','subject'], keep='first')
    df = df.reset_index()
    return df 
 
def time_difference(d1_str, d2_str): 
    """ Given two dates, returning the absoulute time difference in days between them """

    d1 = dt.datetime.strptime(d1_str, "%d/%m/%Y")
    d2 = dt.datetime.strptime(d2_str, "%d/%m/%Y")
    delta = d2-d1
    difference = abs(delta.days)
    return difference   

def keep_first_visit_only(df):
    """ Given a df of visits, keeps only the first visit of a subject in the df """
    
    df = df.copy()
    df.sort_values(by='visit')
    df = df.groupby('subject', as_index=False).first() 
    return df

def initialize_visits(bna, clinical):
    """ Given two paths to both the BNA and clinical data sets, returns a merged df (named 'visits')
    that contains all visits, merged to the same dataframe with both BNA and clinical columns """

    # Step 1: open csv and create the 'final_date' column for each visit, to avoid ambiguity in dates
    reformat_dates(bna)  
    reformat_dates(clinical)
    unite_same_day_visits_to_the_first(bna)  
    # Step 2: merge the two dataframes (BNA and clinical) by 'subject' and 'final_date' of visit
    visits = final_date_merge(bna, clinical,t=2) 
    return visits

def create_dict_by_column_c(df,c): #THIS FUNCTION WAS CALLED get_c_to_group_by_c_dict
    """ Given a df and a column namc c, creates a dictionary that groups keys = c values to vals = dfs of c values """

    df_grouped_by_c = df.groupby([c])
    c_to_c_group = {}
    for name,group in df_grouped_by_c: #map all groups by key= name= subject, val = group = subject's df (for time complexity reasons)
        group.sort_values(by=c)
        c_to_c_group[name] = group
    return c_to_c_group

def add_columns_to_visits(visits,bna, subject_to_subject_group,columns_type:str,use_file:bool):
    """ Adds a column to visit given a string that represents the type of column wanted """

    colums_to_join = None
    if columns_type == "change_visit_2_to_visit_3":
        if use_file: # we use it only when editing code and relying on a ready csv (much faster)
            colums_to_join = pd.read_csv('bna_cols_change_from_visit2_to_visit3.csv')
        else: # when we want to generate change_visit2_visit3 from scratch (much slower- a few minutes)
            colums_to_join = get_electrodes_change_visit2_visit3(visits,bna,subject_to_subject_group)
    # before running the next call, the shape of visits was (182,1500), and after running it,
    # we will get that its shape is (182, 2887)
    visits =  visits.join(colums_to_join) #joining the df 'column_to_join' to the right of visits
    return visits


def get_electrodes_change_visit2_visit3(visits, bna, subject_to_subject_group):
    """ Given visits and BNA df, returns a df that for each subject i and column j calculates the percentage change
    in between the second to the third visit for the subject for column j that belongs to BNA numeric data (electrod values) """

    bna_numeric = bna._get_numeric_data() #only the eeg numeric data, exclude date columns and 'subject' for example
    new_cols_list =[] 
    for col_name in bna_numeric.columns.values:
        new_col_name = col_name + "_change_visit2_visit3"
        new_col = visits['subject'].apply(lambda subject: get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_to_subject_group[subject],c=col_name,i=1,j=2)) #i = 1 means visit 2, j=2 means visit 3
        new_col.rename(new_col_name, inplace=True) #rename new_col to the new name
        new_cols_list.append(new_col) #add the new col to the list of columns
    all_new_cols = pd.concat(new_cols_list, axis=1, ignore_index=False) #concat all new change_visit2_visit3 columns to one df
    return all_new_cols

   
def get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_df,c,i,j):
    """ Given a data frame of all the visits (sorted by visit number) of a particular subject,
    returns the value of the change rate from visit i to visit j in the column named c,
    such that i is at least 0, j is at most number of visits-1 and i < j.
    Note that if the number of last visit is unknown, you can transfer j == "last" and the
    function will convert it into the last index in the df for this perticular subject """

    n = len(subject_df)
    i_visit_val = subject_df.iloc[i][c] if n-1 >= i >= 0 else None #the value of the i'th visit in col c
    j_visit_val = subject_df.iloc[j][c] if n-1 >= j >= 0 else None #the value of the j'th visit in the col c
    if i_visit_val and j_visit_val and i_visit_val != 0:
        percantage_change = ((j_visit_val - i_visit_val) / j_visit_val) * 100 
    else:
        percantage_change =  0 #a deafult value
    return percantage_change


def get_predictive_features(visits,clinical):
    """ Given a df 'visits': 1. drops out all second and later visits for each subject such that each line will represent 
    a subject and 2. drops out all the non predictive columns in visits (only columns that will take part in the model will remain).
    Note that the only column that will remain in visits but won't take part as a feature, is 'subject'.
    All in all, this function returns a df of ['sujbect','predictor1','predictor2'....] """

    # Step 1: only keep the first visit for each subject to be considered tge baseline X values
    # Specifically, we'll have as many rows as the number of subjects
    visits = keep_first_visit_only(visits) 

    # Step 2: change the 'gender' column from 'male', 'female' to 1,0 (linear regression demands numeric values)
    visits = change_string_column_to_numeric_binary(visits,'gender')

    # Step 3: remove non numeric and 'noisy' columns (like dates, visit numbers, etc..) from the df.
    # Step 3.a: pick the 'noisy' and non numeric and\or irellevant for prediction columns
    
    # Note that the next block of code (the picking of columns to drop) is based on the clinical and BNA data frames
    # and their style of column names. When we get new data frames, we need to change this block too.
    clinical_col_set = set(list(clinical.columns.values))
    allowed_cols_clinical = ['subject','Weight in Kg','height in cm ','BMI','Smoking?']
    restricted_cols_clinical = get_set_without_items(clinical_col_set, items = allowed_cols_clinical)
    restricted_cols_bna = set(['taskData.elm_id','EEG NUMBER','visit','date','ageV1'])
    restricted_cols_visits =set(['level_0','index','date_x','date_y','just_date_x','just_date_y','final_date',])

    cols_to_drop = restricted_cols_visits.union(restricted_cols_bna.union(restricted_cols_clinical)) #the set
    # Step 3.b: drop those specific columns from the dataframe
    visits = drop_cols(visits, cols_to_drop) 
    return visits


def change_string_column_to_numeric_binary(df,col_name): 
    #THIS FUNCTION WAS CALLED change_gender instead of change_string, changed it to be more inclusive
    """ Convert a column of strings to a numeric columns, specifically converting "male" and "female"
    from a gender column to 1 and 0 to be used numberically later """
    
    if col_name == "gender": #if the next data set has a different name for the gender column, we'll add it here
        df[col_name] = df[col_name].apply(lambda x: 1 if x == "Male" else 0)
        for col in df.columns:
            if col != "subject":
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_set_without_items(s:set,items:list):
    """ Given a set s and a list of items in it, returns the set without those items """

    for item in items:
        if item in s:
            s.remove(item)
    return s

def drop_cols(df,cols_to_drop:set):
    """ Given a set 'cols_to_drop' of column names, removes the columns in it from the df and returns it """

    df = df[[c for c in df.columns if c not in cols_to_drop]] #allowing only columns we weren't marked to remove
    return df

def generate_X(bna_path, clinical_path): 
    """ This is the main function of this module. It returns a tuple of a df and a dictionary """

    # Step 1: create a dataframe from the two raw datasets named 'visits'
    bna = pd.read_csv(bna_path) 
    clinical = pd.read_csv(clinical_path)
    visits = initialize_visits(bna, clinical) #creates a df named visits from the raw data - a merged version of them by date.
    subject_to_subject_group = create_dict_by_column_c(visits, c='subject') #creates the dictionary of key = subject, val = it's df in visits
    visits = add_columns_to_visits(visits,bna, subject_to_subject_group,columns_type="change_visit_2_to_visit_3",use_file=True)
    subject_to_subject_group = create_dict_by_column_c(visits, c='subject') #updating the dict after editing it
    
    # Step 2: create a new df named 'subjects' in which each subject has one row representing it, which contains only the 
    # predicting features from visits (baseline treatment features, change in BNA features from visit2 to visit3)
    # Important: note that 'subject' is the initial shape of the 'X' vector (before reducing dimensions with select k best
    # inside the model itself. The 'y' predicted value is generated for this X vector inside the model too).
    subjects = get_predictive_features(visits,clinical)
    return (subjects, subject_to_subject_group)

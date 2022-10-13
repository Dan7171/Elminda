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
    """Add documentiaton here- something like: 'Given df, returning the...' """

    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df['just_date'] = pd.to_datetime(df['date']).dt.date
    newdate = dt.date(1998, 11, 8)
    df['just_date'].replace({pd.NaT: newdate}, inplace=True)
    df['just_date'] = df['just_date'].apply(lambda d: d.strftime("%d/%m/%Y"))
    addcol = df.pop('just_date')
    df.insert(2, 'just_date', addcol)

def unite_same_day_visits_to_the_first(df):
    """ for a subject with more than one EEG recording in the same day, keep only the first visit"""
    df = df.sort_values(by=['subject','date']) #sorting visits by subject, and then by date
    df = df.reset_index()
    df = df.drop_duplicates(subset=['subject', 'just_date'], keep='first') #removes rows(visits) who has same subject with the same date, but leaves the first appearance
    df = df.reset_index()

def final_date_merge(df1, df2, t): 
    """merging df1 and df2 by subject and creates a final_date column for each visit. t is the valid maximun difference in days"""
    
    df = df1.merge(df2, how = 'inner',on = ['subject']) #leaving in the df all pairs with the same subject
    # Considering only visits where the bna date of visit and the clinical date of visit have at most
    # t days between them, and uniting them with one 'final_date'.Note that the function is droping duplicates and keeping
    # only the first appearance of a final date, in order to prevent duplication of 'fianl_date's of visits.
    #we want each visit (a pair of clinical visit and bna visit) to have one final_date
    df['final_date'] =[d1 if time_difference(d1,d2)<= t else "remove" for d1,d2 in zip(df['just_date_x'],df['just_date_y'])] 
    df = df[df['final_date']!= 'remove']
    df = df.reset_index()
    df = df.drop_duplicates(subset=['final_date','subject'], keep='first')
    df = df.reset_index()
    return df 
 
def time_difference(d1_str, d2_str): #v
    """Given two dates, returning the absoulute time difference in days between them"""
    d1 = dt.datetime.strptime(d1_str, "%d/%m/%Y")
    d2 = dt.datetime.strptime(d2_str, "%d/%m/%Y")
    delta = d2-d1
    difference = abs(delta.days)
    return difference   

def get_c_to_group_by_c_dict(df,c):
    """Given a df and a column namc c, creating a dictionary of grouping key= c values to vals = dfs of c values"""
    df_grouped_by_c = df.groupby([c])
    c_to_c_group = {}
    for name,group in df_grouped_by_c: # map all groups by key= name= subject, val = group = subject's df (for time complexity reasons)
        group.sort_values(by=c)
        c_to_c_group[name] = group
    return c_to_c_group

def add_columns_to_visits(visits,bna, subject_to_subject_group,columns_type:str,use_file:bool):
    """Given a string represents the type of columns we want to add to visits,
    we will add them """
    colums_to_join = None
    if columns_type == "change_visit_2_to_visit_3":
        if use_file: # we use it only when editing code and relying on a ready csv (much faster)
            colums_to_join = pd.read_csv('bna_cols_change_from_visit2_to_visit3.csv')
        else: # when we want to generate change_visit2_visit3 from scratch (much slower- a few minutes)
            colums_to_join = get_electrodes_change_visit2_visit3(visits,bna,subject_to_subject_group)
    
    #print(visits.shape) -> (182, 1500)
    #print(colums_to_join.shape) -> (182, 1387)
    visits =  visits.join(colums_to_join) # joining the df 'column_to_join' to the right of visits
    #print(ans.shape) -> (182, 2887)
    return visits


def get_electrodes_change_visit2_visit3(visits,bna,subject_to_subject_group):
    """Given visits and bna df, returns a df that for each visit i, column j is the change rate in percents
    in between the second to the third visit for the subject with that visit i, in col j  that
    belongs to bna numeric data (electrod values)"""
    bna_numeric = bna._get_numeric_data() #only the eeg numeric data, exclude date columns and 'subject' for example
    new_cols_list =[] 
    for col_name in bna_numeric.columns.values:
        new_col_name = col_name + "_change_visit2_visit3"
        # calculte for each column from bna_numeric. a new column that represents the change in it
        # from visit 2 to visit 3, for each subject:
        new_col = visits['subject'].apply(lambda subject: get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_to_subject_group[subject],c=col_name,i=1,j=2)) #i = 1 means visit 2, j=2 means visit 3
        new_col.rename(new_col_name, inplace=True) #rename new_col to the new name
        new_cols_list.append(new_col) # add the new col to the list of columns
    all_new_cols = pd.concat(new_cols_list, axis=1, ignore_index=False) #concat all new change_visit2_visit3 columns to one df
    return all_new_cols

   
def get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_df,c,i,j):
    """Given a data frame of all the visits (sorted by visit number) of a particular subject,
    returns the value of the change rate from before to after, in the column named c,
    where the before change is the value of c in the i'th (starts from 0) visit of this subject and
    after change is the value of c in the j'th visit (<= number of visits -1 ) of this subject.
    note that if the number of last visit is unknown, you can transfer j == "last" and the
    function will conver it into the last index in the df of this perticular subject"""
    n = len(subject_df)
    i_visit_val = subject_df.iloc[i][c] if n-1 >= i >= 0 else None #the value of the i'th visit in col c
    j_visit_val = subject_df.iloc[j][c] if n-1 >= j >= 0 else None #the value of the j'th visit in the col c
    
    if  i_visit_val and j_visit_val and i_visit_val != 0:
        percantage_change = ((j_visit_val - i_visit_val) / j_visit_val) * 100 
    else:
        percantage_change =  0 # a deafult value
    return percantage_change

def initialize_visits(bna, clinical):
    """Given two pathes to the bna (eeg mostly) and clinical data sets, returns  a merged df (named 'visits')
    which contains all visits, merged to same date with both bna and clinical columns"""
    #Step 1: open csvs and create the 'final_date' column for each visit, to avoid ambiguity in dates:
    reformat_dates(bna)  
    reformat_dates(clinical)
    unite_same_day_visits_to_the_first(bna)  
    #Step 2: merge the two data frames (bna and clinical) by 'subject' and 'final_date' of visit
    visits = final_date_merge(bna, clinical,t=2) 
    return visits

def get_predictive_features(visits,clinical):
    """Given a df 'visits': 1. dropping out all second or later visits for each subject in order that
    each line will represent a subject and 2. drops out all the non predictive columns in visits (only columns that will take part in the model will remain)
    the only column that will remain in visits but not taking part as a feature, is 'subject'
    returning ['sujbect','predictor1','predictor2'....]
    """

    #Step 1: keeps first visit of each subject and only it (as 'baseline' 'X' values)
    # for example, for 40 subjects, we will remain with 40 visits (rows) only
    visits = keep_first_visit_only(visits) 
    #Step 2: changes 'gender' from 'male','female' to 1,0 (linear regression demands numeric values):
    visits = change_gender_column_to_numeric_binary(visits,'gender')# turning 'female' to 0 and 'male' to 1 under 'gender' column:
    

    #step 3 : removes non numeric and 'noisy' columns (like dates, visit numbers etc..) from the df:
    #prepare one big set of them:
    #note: the next block of code (the picking of columns to drop) is based on the clinical and bna data frames
    # and their style of coulum names. when we get new data frames, we need to change this block too.
    #3.1 :pick the 'noisy' and non numeric and\or irellevant for prediction columns:
    
    clinical_col_set = set(list(clinical.columns.values))
    allowed_cols_clinical = ['subject','Weight in Kg','height in cm ','BMI','Smoking?']
    restricted_cols_clinical = get_set_without_items(clinical_col_set, items = allowed_cols_clinical)
    restricted_cols_bna = set(['taskData.elm_id','EEG NUMBER','visit','date','ageV1'])
    restricted_cols_visits =set(['level_0','index','date_x','date_y','just_date_x','just_date_y','final_date',])
    cols_to_drop = restricted_cols_visits.union(restricted_cols_bna.union(restricted_cols_clinical)) #the set
    #3.2 : drop them:
    #print("before step 3, shape =", df_copy.shape)
    visits = drop_cols(visits, cols_to_drop) 
    #print("after step 3, shape =", df_copy.shape)
    return visits

def keep_first_visit_only(df):
    """Given df of visits, saves only the first visit of a subject in the df"""
    df = df.copy()
    df.sort_values(by='visit')
    df = df.groupby('subject', as_index=False).first() #keeps only the first visit of each subject   
    #print(df)
    return df

def change_gender_column_to_numeric_binary(df,gender_col_name):

    """convert gender: 'male'/'female' -> 0/1 (so we can use it numericaly) under the couln named gender_col_name"""

    df[gender_col_name] = df[gender_col_name].apply(lambda x: 1 if x == "Male" else 0)
    for col in df.columns:
        if col != "subject":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_set_without_items(s:set,items:list):
    """Given a set s and a list of items in it, returns a new set without those items"""
    for item in items:
        if item in s:
            s.remove(item)
    return s

def drop_cols(df,cols_to_drop:set):
    """Given a set 'cols_to_drop' of column names, removing the columns in it from the df and returns it"""
    df = df[[c for c in df.columns if c not in cols_to_drop]] # allowing only columns we weren't marked to remove
    return df

def generate_X(bna_path, clinical_path): #returns a tuple of df and dict
    """Main function of this module"""
    bna = pd.read_csv(bna_path) 
    clinical = pd.read_csv(clinical_path)
    visits = initialize_visits(bna,clinical) # from raw data (bna and clinical dfs), creates a df named visits, a merged version of them by date.
    subject_to_subject_group = get_c_to_group_by_c_dict(visits,c='subject')  # creates the dictionary of key = subject, val = it's df in visits
    visits = add_columns_to_visits(visits,bna, subject_to_subject_group,columns_type="change_visit_2_to_visit_3",use_file=True)
    subject_to_subject_group = get_c_to_group_by_c_dict(visits,c='subject') #updating the dict after editing it
    # STEP 2 Filter out(drop) from visits all the non-predictive columns except 'subject' (like dates, names and so on...):
    #CREATE A NEW DF NAMED 'subjects'. in 'subjects' there will be ONE line for each subject, only with predicting features from visits (baseline treatment start features, change in bna features from visit2 to visit3)
    #important: note that 'subject' is the initial shape of the 'X' vector (before reducing dimensions with select k best inside the model itself. the 'y' predicted value is generated for this X vector inside the model too)
    subjects = get_predictive_features(visits,clinical)  # leaves in visits only one row for each subject (by first visit) relevant numeric columns  + the column 'subject' (the only categorial column)
    return (subjects, subject_to_subject_group)

 
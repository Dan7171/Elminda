import array
from heapq import merge
from itertools import groupby
from locale import format_string
from pathlib import Path
from statistics import correlation
from time import strftime
from matplotlib import pyplot as plt, test
from pyparsing import trace_parse_action

from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_selection import chi2
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


def final_date_merge(df1, df2): 
    """merging df1 and df2 by subject and creates a final_date column for each visit """
    t = 2 #t is the valid maximun difference in days
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
    d1 = dt.datetime.strptime(d1_str, "%d/%m/%Y")
    d2 = dt.datetime.strptime(d2_str, "%d/%m/%Y")
    delta = d2-d1
    difference = abs(delta.days)
    return difference   

def save_df_to_csv(name, df):
    path = Path()
    df.to_csv((name + '.csv'),index = False)


def get_percantage_change(df, column_name):
    pass
     

def group_by(df,c): #visits.groupby(['subject']), #df.groupby([c])
    """ equivalent to df.groupby([c]), I wrote this func for exampling the outputs!

    given a df (visits) returns all of the sub dfs, where each sub df belongs so one subjects:
    sub44
        visit  final_date
    46      1  22/08/2016
    47      2  29/08/2016
    48      3  09/05/2016
    49      5  20/09/2016
    50      6  27/09/2016
    sub83
        visit  final_date  HDRS-17
    53      1  12/11/2016     23.0
    54      2  18/12/2016     22.0
    55      3  25/12/2016     21.0
    56      4  01/01/2017     21.0
    57      5  01/08/2017     20.0
    58      6  16/01/2017     14.0
    .
    .
    ."""
    return df.groupby([c])
    
 

def get_group_by_name_from_all_groups(groups_by_coulumn,desired_group_name):
    """first getting all athe groups by 'subject'picks particulary the group of the subject passed in subject_id
    example: input = (groups_by_coulumn=visits.groupby['subject] (all groups by subjects),desired_group_name='sub83') -> output = :
    
    output = sub83,
    sub83
        visit  final_date  HDRS-17 .....
    53      1  12/11/2016     23.0
    54      2  18/12/2016     22.0
    55      3  25/12/2016     21.0
    56      4  01/01/2017     21.0
    57      5  01/08/2017     20.0
    58      6  16/01/2017     14.0"""
    
    for name, group in groups_by_coulumn: #name is group name('subject'), group is all its df
        if name == desired_group_name:
            return group # a df
            #print(name)
            #print(group[['visit','final_date']]) #see output example in ducumentation

def get_subject_first_visit_val_in_column_c(subject_id,d,c,):
    group = d[subject_id] # O(1)
    #group.sort_values(by='visit')
    return group.iloc[0][c]
    
def get_subject_last_visit_val_in_column_c(subject_id,d,c):
    group = d[subject_id] # O(1)
    #group.sort_values(by='visit')
    return group.iloc[len(group)-1][c]

def print_interesting_values_by_subject_grouping(df,d):
    """adding desired priniting columns which are unneccessary in model (like first_visit_HDRS-17),
    printing grouping by subject of all wanted featured and then deleting the added unnccessary ones"""
    
    df['first_visit_HDRS-17'] = df['subject'].apply(lambda subject: get_subject_first_visit_val_in_column_c(subject,d=subject_to_subject_group,c='HDRS-17'))
    df['last_visit_HDRS-17'] = df['subject'].apply(lambda subject: get_subject_last_visit_val_in_column_c(subject,d=subject_to_subject_group,c='HDRS-17'))
    #after addign unnccessary columns: print groups by subject:
    for name,group in group_by(df,'subject'):
        print(name)
        print(group[['visit','final_date','HDRS-17']])
        print(group[['first_visit_HDRS-17','last_visit_HDRS-17','change_HDRS-17']])
    #remove unneccessary
    
    df = df.drop(columns=['first_visit_HDRS-17','last_visit_HDRS-17'])
    



def get_subjects_with_increase_in_HDRS_or_with_zero_start_HDRS(visits):
    """returning all the strange subjects that has HDRS-17 == 0 in the
    first visit or had a positive percantage change in change_HDRS-17 (increased depression)"""
    
    strange_subjects = set(visits.iloc[i]['subject'] for i in range(len(visits)) if (visits.iloc[i]['change_HDRS-17'] == None or float(visits.iloc[i]['change_HDRS-17']) > 0))
    return strange_subjects
    
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

def select_features_by_univariate_selection(df): 
    "by chi square, anova test, or corellation coefficient(pearson or spirman)?"
    "we will use the method selectKbest- which finds the K best features in terms of maximun corellation with the y variant"    
    pass
def seletct_features_by_feature_importance(df): #all 3 technices are usefull for small data sets only. not usefull for us.
    "part of the wraper methods"
    "USEFUL ONLY IN SMALL DATA SETS"
    "by wraper method. Some of them with no statistical tasks, but only one of 3 mechanismes:"
    "forward selection/backward elimination/recursion feature elimination. forward selection- start with no features, adding only highly corellated with y features to the model.  backward elimination: start with all of features, find the least significant feature on y, if its p val >0.05 drop it else leave it,"
    "and move on to the next feature untill end of features"
    pass
def select_features_by_correlation_matrix_heat_map(df):
    pass
def change_gender_column_to_numeric_binary(df,gender_col_name):
    """convert gender: 'male'/'female' -> 0/1 (so we can use it numericaly) under the couln named gender_col_name"""

    df[gender_col_name] = df[gender_col_name].apply(lambda x: 1 if x == "Male" else 0)
    for col in df.columns:
        if col != "subject":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
def keep_first_visit_only(df):
    df.sort_values(by='visit')
    df = df.groupby('subject', as_index=False).first() #keeps only the first visit of each subject   
    return df

def drop_cols(df,cols_to_drop:set):
    """Given a set 'cols_to_drop' of column names, removing the columns in it from the df and returns it"""
    
    df = df[[c for c in df.columns if c not in cols_to_drop]] # allowing only columns we weren't marked to remove
    return df
def get_set_without_items(s:set,items:list):
    """Given a set s and a list of items in it, returns a new set without those items"""
    for item in items:
        if item in s:
            s.remove(item)
    return s

def print_conclusions(model_name,predicted_value,score,data_size,features_number):
    """Given a tested model, supplying conclusion of the train and prediction"""

    print("MODEL NAME: ", model_name)
    print("DATA SIZE (TOTAL OBSERVATIONS NUMBER- TRAIN + TEST): ", data_size)
    print("FEATURES NUMBER: ", features_number) #size of X vector
    print("PREDICTED VALUE: ", predicted_value) # y column name
    if model_name == "lr":
        score_type = 'r2_score'
        if score < 0.5:
            print("SCORE TYPE: ",score_type)
            print(f"SCORE: {score}, too low (the closer to 1, the better the model is)")
        else:
            print("SCORE = ", score)

 

def get_electrods_change_visit2_visit3(visits,bna,subject_to_subject_group):
    """Given visits and bna df, returns a df that for each visit i, column j is the change rate in percents
    in between the second to the third visit for the subject with that visit i, in col j  that
    belongs to bna numeric data (electrod values)"""
    bna_numeric = bna._get_numeric_data()
    new_cols_list =[] 
    print(bna_numeric)
    for col_name in bna_numeric.columns.values:
        new_col_name = col_name + "_change_visit2_visit3"
        new_col = visits['subject'].apply(lambda subject: get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_to_subject_group[subject],c=col_name,i=1,j=2)) #i = 1 means visit 2, j=2 means visit 3
        new_col.rename(new_col_name, inplace=True)
        new_cols_list.append(new_col)
    all_new_cols = pd.concat(new_cols_list, axis=1, ignore_index=False)
    return all_new_cols

def lr(visits,subject_to_subject_group,bna,clinical,k_select_best_k):
    #PART A: pick a 'y' to predict and add it to the df
    # preparing linear regression 'y' vector: ('change_HDRS-17') 
    visits['change_HDRS-17'] = visits['subject'].apply(lambda subject: get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_to_subject_group[subject],c='HDRS-17',i=0,j=len(subject_to_subject_group[subject])-1))
    visits = visits[visits['change_HDRS-17']!= 0] #drop out subjects with no change (meaning they have problems in data)
    #PART B: generate 'X' vector, by dropping out columns and rows we can't or don't want to rely on when predicting:
    # preparing linear regression 'X' vector:
    visits_lr = visits

    #step 1: keep first visit of each subject and only it (as 'baseline' 'X' values)
    # for example, for 40 subjects, we will remain with 40 visits (rows) only
    visits_lr = keep_first_visit_only(visits_lr) 

    #step 2: change 'gender' from 'male','female' to 1,0 (linear regression demands numeric values):
    visits_lr = change_gender_column_to_numeric_binary(visits_lr,'gender')# turning 'female' to 0 and 'male' to 1 under 'gender' column:

    #step 3: remove non numerical and 'noisy' columns (like dates, visit numbers etc..) from the df:
    #prepare one big set of them:
    #3.1 pick the 'noisy' and non numerical columns:
    #print("before step 3, shape =", visits_lr.shape)
    clinical_col_set = set(list(clinical.columns.values))

    allowed_cols_clinical = ['Weight in Kg','height in cm ','BMI','Smoking?']
    restricted_cols_clinical = get_set_without_items(clinical_col_set, items = allowed_cols_clinical)

    restricted_cols_bna = set(['taskData.elm_id','EEG NUMBER','visit','date','ageV1'])
    restricted_cols_visits =set(['subject', 'level_0','index','date_x','date_y','just_date_x','just_date_y','final_date',])
    cols_to_drop = restricted_cols_visits.union(restricted_cols_bna.union(restricted_cols_clinical)) #the set
    #3.2: drop them:
    visits_lr = drop_cols(visits_lr, cols_to_drop) 
    #print("after step 3, shape =", visits_lr.shape)
    # parts E-H: splitting, feature selection, training, predicting and testing

    X = visits_lr.drop(columns= ['change_HDRS-17'])
    y = visits_lr['change_HDRS-17']

    X.fillna(0, inplace = True)
    y.fillna(0, inplace = True)

    
    #PART E: select k best features from the 'X' vector, by the 'MRMR' (maximun relevancy, minimun redundancy) principal:
    #for the actual machine learning model:

    select = SelectKBest(score_func=f_regression,k=k_select_best_k) #'best' = have the highest correllation with 'y' comparing to others.
    X_new = select.fit_transform(X,y) # besk k features from X, without names
    cols = select.get_support(indices=True)
    X_new = X.iloc[:,cols] #best k features of X, with their original names

    X_new_y = X_new.join(y) # best k features and y (in order to comput corellation between each one of them and the y)
    abs_correlations = abs(X_new_y.corr()['change_HDRS-17'])
    abs_correlations.rename('abs_corr_with_change_HDRS-17', inplace=True)
    print(abs_correlations)

    fig_dims = (10,5)
    fig,ax = plt.subplots(figsize=fig_dims)
    sns.heatmap(X_new_y.corr(),ax=ax)
    plt.draw()


    # PART F: split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
    # PART G: train model with fit

    regressor = LinearRegression()
    trained_model_reg = regressor.fit(X_train, y_train) 

    # PART H: test your model on 
    # r**2 = [1- (ss(res)-ss(total)]. ss(res) = sum of squares between y_pred (regression line ys for X_test) minus y_test, ss(total)= sum of squares between average y_train minus y_test
    # the rule is:  the higher r2_score (<=1), the more accurate our model is
    r2_score = regressor.score(X_test,y_test) #predicting X_test on the model, and calculating r**2 score
    print_conclusions(model_name= "lr", predicted_value = "change_HDRS-17", score=r2_score,data_size = X_new.shape[0],features_number = X_new.shape[1])
    print("best K features: ",X_new.columns.values)
    print("finished running with no bugs")
    plt.show()

def knn(visits,subject_to_subject_group,bna,clinical,k_select_best_k,k_knn):
    pass
    # 
    # do for classification:
    #

    # y = visits['HDRS_drop_level'] # level is one of 3 labels: ,'respone','non-response',and 'remission'
    # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    #knn = KNeighborsRegressor(n_neighbors=3) 3 for response, non- response and remission
    #knn.fit(X_train, y_train)
    #pred = knn.predict(X_test)
    #mae = mean_absolute_error(y_test, pred) , to compute the mean absolute error (optional)


def main(model_name:str,bna_path:str,clinical_path:str,k_select_best_k=None, k_knn=None):
    """Given a model name from {'lr'/'knn'...},two paths to clinical and bna data csv files,
    optional k for select_best_k, optional k for knn , training and testing the model in model_name on the arguments """
    #*** MAIN PART : ***
    #PART A: Prepare data for work:
    #A.1 open csvs and create the 'final_date' column for each visit, to avoid ambiguity in dates:
    bna = pd.read_csv(bna_path) 
    clinical = pd.read_csv(clinical_path) 
    reformat_dates(bna)  
    reformat_dates(clinical)
    unite_same_day_visits_to_the_first(bna)  
    #A.2 merge the two data frames (bna and clinical) by 'subject' and 'final_date' of visit
    visits = final_date_merge(bna, clinical) 
    #A.3 create a dictionary of key = 'subject' to val = all the df values belong to it (for time complexity reasons):
    visits_grouped_by_subject = group_by(visits,'subject') # all groups by subject, same as df.groupby(['subject])
    subject_to_subject_group = {}
    for name,group in visits_grouped_by_subject: # map all groups by key= name= subject, val = group = subject's df (for time complexity reasons)
        group.sort_values(by='visit')
        subject_to_subject_group[name] = group
    
    #A.4 add for each bna column in visits (columns of electrodes), the change rate in percents between visit 2 (before first treatment) to visit 3 (after first treatment)
     
    #THE NEXT CALL IS THE RIGHT WAY TO GENERATE THE NEW DF, BUT SUPER HEAVY (A FEW MINS):
    #bna_columns_change_visit2_visit3 = get_electrode_visit2_visit3_change(visits,bna,subject_to_subject_group)
    #INSTEAD, I SAVED THE DF AFTER GENERATING IT ONCE, SO IT WON'T TAKE THAT LONG TO PRODUCE IT EVERY TIME:
    bna_columns_change_visit2_visit3 = pd.read_csv('bna_cols_change_from_visit2_to_visit3.csv')
    #save_df_to_csv('bna_cols_change_from_visit2_to_visit3',all_new_cols)
    visits = visits.join(bna_columns_change_visit2_visit3)
    
    #PART B: pick a model and train it on the data: 
    if model_name == "lr":
        lr(visits,subject_to_subject_group,bna,clinical,k_select_best_k)
    if model_name == "knn":
        knn(visits, subject_to_subject_group, bna, clinical,k_select_best_k, k_knn)

     
# IMBALANCED DATA SET --- WE DO THIS ONLY IF OUR ML ISN'T RELIABLE
# Below is an example for undersampling - a technique to help the ML
# be more reliable when the data isn't ideally distributed,
# when "data" is the name of the final df we're working on:

# shuffled = data.sample(frac=1,random_state=4)

# # Put all the "smaller group" (which is the group that takes up the majority of the data) in a separate dataset.
# smaller_group = shuffled.loc[shuffled['Class'] == 1]

# #Randomly select n observations from the majority class
# bigger_group = shuffled.loc[shuffled['column'] == 0].sample(n=n,random_state=42)

# # Concatenate both dataframes again
# normalized = pd.concat([smaller_group, bigger_group])

# #plot the dataset after the undersampling, this part might not be the same for us
# plt.figure(figsize=(8, 8))
# sns.countplot('Class', data=normalized)
# plt.title('Balanced Classes')
# plt.show()


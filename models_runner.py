import array
from heapq import merge
from itertools import groupby
from locale import format_string
from pathlib import Path
from matplotlib import pyplot as plt, test
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.neighbors import kneighbors_graph
from X_initializer import create_dict_by_column_c, get_subject_change_rate_in_column_c_from_visit_i_to_visit_j
from mrmr import mrmr_classif
from sklearn.datasets import make_classification
warnings.simplefilter("ignore", UserWarning)
#import the_module_that_warns

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

 
#Wasn't sure if we need these or what to write about them, leaving them as is for now:
def select_features_by_univariate_selection(df): 
    "by chi square, anova test, or correlation coefficient(pearson or spirman)?"
    "we will use the method selectKbest- which finds the K best features in terms of maximun correlation with the y variant"    
    pass
def select_features_by_feature_importance(df): #all 3 technices are usefull for small data sets only. not usefull for us.
    "part of the wraper methods"
    "USEFUL ONLY IN SMALL DATA SETS"
    "by wraper method. Some of them with no statistical tasks, but only one of 3 mechanismes:"
    "forward selection/backward elimination/recursion feature elimination. forward selection- start with no features, adding only highly correlated with y features to the model.  backward elimination: start with all of features, find the least significant feature on y, if its p val >0.05 drop it else leave it,"
    "and move on to the next feature untill end of features"
    pass
 

def print_conclusions(model_name,K_best_features,y_name, score, data_size,features_number):
    """ Given a tested model, supplies conclusions of the train and prediction """
    print("______________________________________________________________")
    print("*** CONCLUSIONS OF MODEL TRAINING AND TESTING: ***\n")
    print("MODEL NAME:", model_name)
    print("DATA SIZE (TOTAL OBSERVATIONS NUMBER: TRAIN + TEST):", data_size)
    print("'X' BEST K FEATURES NUMBER:", features_number) #the size of X vector
    print("'X' BEST K FEATURES:",'\n',K_best_features)
    print("PREDICTED 'y' VALUE:", y_name) #the 'y' column name

    if model_name == "lr":
        #Score function in lr is 'r2_score' (r squared).
        score_func = "r2_score (r squared)"
        print("SCORE FUNCTION: ", score_func)
        if score < 0.5:
            print(f"SCORE: {score}, too low (the closer to 1, the better the model is)")
        else:
            print("SCORE = ", score)

    if model_name  == 'knn':
         #Score func in knn is 'knn.score'. It calculates the ratio of correct predictions to all predictions.
        score_func = 'knn_score - ratio of #correct predictions (y_predict) / #all right values (y_test)'
        print("SCORE FUNCTION:", score_func)
        print("SCORE:",str(score*100)+'%'+" ACCURACY IN PREDICTION")
    
    print("______________________________________________________________")    

def get_subject_end_of_treatment_state_in_column_c(subject_df,c):
    """Given a df of a particular subject (in sorted order of visits) and a column name c, 
    returns the end-of-experiment value for this subject in column c """

    num_of_visits = len(subject_df)
    ans = subject_df.iloc[num_of_visits-1][c] #the value of the last visit in the column named c
    return ans

def convert_HDRS_17_score_to_class(HDRS_17_score,HDRS_17_change_rate):
    """ The subjects will be divided into three clusters according to their HDRS-17 score, such that
    if their final HDRS-17 score is 7 or less they'll be considered 'remission', if the rate change
    between their first and last visits shows more than 50% improvement they'll be considered 'responsive',
    otherwise we will consider them 'non-responsive' """

    ans = None
    if 0 <= HDRS_17_score <= 7:
        ans = "Remission" #also called normal state
    else:
        if HDRS_17_change_rate < -50:
            ans = "Responsive"
        else:
            ans = "Non-responsive"
    return ans

def filling(df):
    """ Fill NaN cells with zeroes for numeric models. This function was created to prevent
    code duplications in multiple places """
    df.fillna(0, inplace = True)

def selector_func(X, y, k_select_k_best, funcname):
    # Step 1: selector will get us the k features with the highest correlation with 'y'
    selector = SelectKBest(score_func=funcname,k=k_select_k_best)
    # Step 2: get the best k features from X without the columns names
    X_new = selector.fit_transform(X,y)
    # Step 3: get the indices of the k best features' columns
    cols = selector.get_support(indices=True)
    # Step 4: get the actual names of the k best features, and join them into one df to compute correlation between them
    X_new = X.iloc[:,cols]
    X_new_y = X_new.join(y)
    return X_new, X_new_y
 
def lr(subject_to_subject_group,subjects_X,k_select_k_best, y_name):
    """ Training and reporting scoring results for the linear regression model """

    # Step 1: create the y column for the dataframe, then remove it and the subject column to get the final X columns
    # so that at the end of this step we'll have X and y ready
    if y_name == 'change_HDRS-17':
        subjects_X[y_name] = subjects_X['subject'].apply(lambda subject: get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_to_subject_group[subject],c='HDRS-17',i=0,j=len(subject_to_subject_group[subject])-1))
        subjects_X = subjects_X[subjects_X[y_name]!= 0] #drop out subjects with no change (meaning there's a problems with the data)
    X = subjects_X.drop(columns= ['subject',y_name]) 
    # For later: note that at the moment we drop the subject column right before training the model. This is not the most 
    # beaufitul design and it was caused by the 'subejct_to_subject_group' dictionary. We can somehow prevent this 
    # by maybe changing the df 'visits' (subjects_X in this case) to a new df with row == subject (without cols like 
    # subject number or name which is redundant) and and columns will be best k features (relevant for any model) and the y column.
    y = subjects_X[y_name]
    filling(X)
    filling(y)
    
    # Step 2: select k best features from the 'X' vector, by the 'MRMR' (maximun relevancy, minimun redundancy) principal:
    X_new, X_new_y = selector_func(X, y, k_select_k_best, f_regression)
    
    # This is where I tried to make the MRMR work but the statistics are not as good as without it:
    # aftermrmr = mrmr_classif(X, y, K = k_select_k_best)
    # X_new1 = X[[c for c in X.columns if c in aftermrmr]]
    # X_new_y1 = X_new1.join(y)
    # abs_correlations1 = abs(X_new_y1.corr()[y_name])
    # abs_correlations1.rename('abs_corr_with_y', inplace=True)
    # print("top k highest features in correlation to y and their abs correlations1:")
    # print(abs_correlations1)

    # not a step, but saving and showing plots and correlations:
    abs_correlations = abs(X_new_y.corr()[y_name])
    abs_correlations.rename('abs_corr_with_y', inplace=True)
    print("top k highest features in correlation to y and their abs correlations:")
    print(abs_correlations)
    fig_dims = (10,5)
    fig,ax = plt.subplots(figsize=fig_dims)
    sns.heatmap(X_new_y.corr(),ax=ax) #create correlations hit map
    plt.draw()

    # Step 3: split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

    # Step 4: train model with fit
    regressor = LinearRegression()
    trained_model = regressor.fit(X_train, y_train) 

    # Step 5: predict X_test y scores (y_prediction) and compare them to real y scores (y_test)
    r2_score = regressor.score(X_test,y_test) #doing both prediction and r2 calculation
    print_conclusions(model_name= "lr",K_best_features=X_new.columns.values, y_name = y_name,score=r2_score,data_size = X_new.shape[0],features_number = X_new.shape[1])
    print("finished running with no bugs")
    plt.show() #entering an infinite while loop

  
def knn(subject_to_subject_group,subjects_X,k_select_k_best,k_knn, y_name):
    """ Training and reporting scoring results for the knn model """

    # Step 1: create the y column for the dataframe, then remove it and the subject column to get the final X columns
    # so that at the end of this step we'll have X and y ready
    # Note that the same comment about this step that is written in the lr function is also relevant here
    if y_name == 'end_of_treatment_class_HDRS-17':
        subjects_X['change_HDRS-17'] = subjects_X['subject'].apply(lambda subject: get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_to_subject_group[subject],c='HDRS-17',i=0,j=len(subject_to_subject_group[subject])-1))
        d =  create_dict_by_column_c(subjects_X,'subject')
        subjects_X[y_name] = subjects_X['subject'].apply(lambda subject: convert_HDRS_17_score_to_class(get_subject_end_of_treatment_state_in_column_c(subject_to_subject_group[subject],c='HDRS-17'),get_subject_end_of_treatment_state_in_column_c(d[subject],c='change_HDRS-17')))
        X = subjects_X.drop(columns= ['subject',y_name]) 
    y = subjects_X[y_name]
    filling(X)
    filling(y)

    #Step 2: select k best features from the 'X' vector, by the 'MRMR' (maximun relevancy, minimun redundancy) principal:
    #X_new, X_new_y = selector_func(X, y, k_select_k_best, f_classif) 
    # MAYA'S WORKING ON MRMR HERE
    aftermrmr = mrmr_classif(X, y, K = k_select_k_best)
    X_new = X[[c for c in X.columns if c in aftermrmr]]
    X_new_y = X_new.join(y)

    # Step 3: split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

    # Step 4: train model with fit
    classifier = KNeighborsClassifier(n_neighbors=k_knn)
    trained_model= classifier.fit(X_train, y_train) 

    # Step 5: predict X_test y scores (y_prediction) and compare them to real y scores (y_test)
    knn_score = classifier.score(X_test,y_test) #return the mean accuracy on the given test data and labels
    y_prediction = trained_model.predict(X_test)
    print("prediction of model values: \n", y_prediction)
    print("real class values (y_test): \n ", y_test)
    print_conclusions(model_name= "knn", K_best_features= X_new.columns.values,y_name = y_name, score=knn_score,data_size = X_new.shape[0],features_number = X_new.shape[1])
    print("finished running with no bugs")
   


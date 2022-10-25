import X_initializer
import models_runner

class mainCaller():
    def func(self):
        bna_path = 'BNA_data.csv'
        newdata = 'EYEC.csv'
        clinical_path = "Clinical_data_hebrew.csv"
        # Step 1: prepare X for model training from the input dataframes
        subjects_X, subject_to_subject_group = X_initializer.generate_X(newdata,clinical_path) #initial X df
        # Step 2: copy the dataframe and send it to the wanted model
        X_lr = subjects_X.copy()
        curr_score = models_runner.lr(subject_to_subject_group, X_lr, k_select_k_best=8,y_name='change_HDRS-17')
        X_knn = subjects_X.copy()
        #curr_score = models_runner.knn(subject_to_subject_group, X_knn, k_select_k_best = 30,k_knn = 3,y_name ='end_of_treatment_class_HDRS-17')
        X_dtree = subjects_X.copy()
        #models_runner.dtree(subject_to_subject_group, X_dtree, k_select_k_best = 10, y_name = 'response_to_treatment')
        #print(curr_score)
        return curr_score

mainCaller().func() 

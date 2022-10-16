import X_initializer
import models_runner
bna_path = 'BNA_data.csv'
clinical_path = "Clinical_data_hebrew.csv"
# prepare X for model training
subjects_X, subject_to_subject_group = X_initializer.generate_X(bna_path,clinical_path) #initial X df
# Step 2: RUN MODELS AS YOU WISH
X_lr = subjects_X.copy()
models_runner.lr(subject_to_subject_group, X_lr, k_select_k_best=10,y_name='change_HDRS-17')
X_knn = subjects_X.copy()
#models_runner.knn(subject_to_subject_group, X_knn, k_select_k_best = 40,k_knn = 3,y_name ='end_of_treatment_class_HDRS-17')

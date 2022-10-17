from matplotlib.cbook import violin_stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from main_caller import mainCaller 
# df = sns.load_dataset("penguins")
# clinical = pd.read_csv('clinical_data_hebrew.csv')
# print(clinical)
# clinical_g = clinical.sort_values(by=['subject'])
# print(clinical_g)
# #clinical_g.sort_values(by='visit')S
# #print(clinical_g[['subject','visit']])

sum = 0
for i in range(100):
    sum += mainCaller().func()
print("in average, the score is", sum/100)

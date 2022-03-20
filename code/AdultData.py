
### Importing the general libraries we shall use in this notebook
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting library that use matplot in background
import matplotlib.pyplot as plt # to plot some parameters in seaborn
from DataWrangling import DataWrangling

class AdultData(DataWrangling):
    def __init__(self):
        print('AdultData')
        super().__init__()
    
    def plot(self):
        sns.set_context('talk', font_scale=.9)
        # Box plot helps us see the mean value of a category "Sex" per "Age" in our dataset
        sns.catplot(data = self.dataset, x = 'sex', y = 'education.num', kind = 'box')
        plt.show()
    
    def plotHist(self):
        super().plot()
    
    def encoder(self, column_name, exclude_col = False):
        merged_df = self.dataset_ready.merge(pd.get_dummies(self.dataset_ready[column_name], drop_first=False, prefix=column_name), left_index=True, right_index=True)
        if exclude_col:
            del merged_df[column_name] # Exclude the original column
        return merged_df
        
if __name__ == '__main__':
    
    path = 'Data/Datasets/Input/adult.csv'
    oAdult = AdultData()
    oAdult.path = path
    oAdult.readData()
    oAdult.dataInfo()
    oAdult.plot()
    oAdult.plotHist()
    oAdult.correlation()
    
    
    
    interval = [16, 25, 35, 60, 120]
    groups = ['Young Adult', 'Adult', 'Senior', 'Elder']
    oAdult.setAgeByGroup(interval, groups, 'age','AgeGroup')
    
    category_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country','income', 'AgeGroup']
    oAdult.removeNoise(category_features, '?', 'no_info')
    
    # Dealing with Normalization
    oAdult.applyMinMaxNorm('age')
    oAdult.plotHist()
    
    
    oAdult.dataset_ready = oAdult.dataset.copy()
    category_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country','income', 'AgeGroup'] category_features:
        oAdult.dataset_ready = oAdult.encoder(cat, exclude_col = True)
    
    print(oAdult.dataset_ready.columns)
    
    oAdult.savetoCVS(r'Data/Datasets/Output/AdultDataset.csv');
    
#     print(oAdult.dataset[oAdult.dataset.isna().any(axis=1)])
#     print(oAdult.dataset['age'].unique())
    

    


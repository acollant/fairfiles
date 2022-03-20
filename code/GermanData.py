
### Importing the general libraries we shall use in this notebook
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting library that use matplot in background
import matplotlib.pyplot as plt # to plot some parameters in seaborn
from DataWrangling import DataWrangling


class GermanData(DataWrangling):
    def __init__(self):
        print('GermanData')
        super().__init__()
        
    def plot(self):
        sns.set_context('talk', font_scale=.9)
        # Box plot helps us see the mean value of a category "Sex" per "Age" in our dataset
        sns.catplot(data = self.dataset, x = 'Sex', y = 'Age', kind = 'box')
        plt.show()
    
    def plotHist(self):
        super().plot()
    
    # Check the files available in the data folder
    def dir(self):
        for dirname, _, filenames in os.walk(self.path):
            for filename in filenames:
                print(os.path.join(dirname, filename))
    
    def encoder(self, column_name, exclude_col = False):
       merged_df = self.dataset_ready.merge(pd.get_dummies(self.dataset_ready[column_name], drop_first=False, prefix=column_name), left_index=True, right_index=True)
       if exclude_col:
           del merged_df[column_name] # Exclude the original column
       return merged_df
    
    
if __name__ == '__main__':
    
    #path = 'https://raw.githubusercontent.com/acollant/fairfiles/main/german_credit_data_risk.csv'
    path = 'Data/Datasets/Input/german_credit_data.csv'
    oGerman = GermanData()
    oGerman.path = path
    
    oGerman.dir()

    oGerman.readData(False)
    oGerman.dataInfo()
    oGerman.plot()
    oGerman.plotHist()
    
    #oGerman.drop_(['Job'])
    oGerman.correlation()
    
    interval = (18, 25, 35, 60, 120)
    groups = ['Young Adult', 'Adult', 'Senior', 'Elder']
    oGerman.setAgeByGroup(interval, groups, 'Age','AgeGroup')
    
    # Dealing with Missing values of Saving account and Checking account

    category_features = ['Sex', 'Housing', 'Saving accounts','Checking account','Purpose','Risk']
    oGerman.removeNaN(category_features)
    
    print(oGerman.dataset[oGerman.dataset.isna().any(axis=1)])
    print(oGerman.dataset['Age'].unique())
    
     # Dealing with Normalization
    oGerman.applylog10Norm(['Credit amount'])
    oGerman.applyMinMaxNorm('Age')#,'Age Norm')
    
    
    
    oGerman.dataset_ready = oGerman.dataset.copy()
    
    category_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk', 'AgeGroup']
    for cat in category_features:
        oGerman.dataset_ready = oGerman.encoder(cat, exclude_col = True)
    
    print(oGerman.dataset_ready.columns)
    
    oGerman.savetoCVS(r'Data/Datasets/Output/GermanDataset.csv');
    
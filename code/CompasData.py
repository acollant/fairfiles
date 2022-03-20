
### Importing the general libraries we shall use in this notebook
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting library that use matplot in background
import matplotlib.pyplot as plt # to plot some parameters in seaborn
from DataWrangling import DataWrangling


class CompasData(DataWrangling):
    def __init__(self):
        print('CompasData')
        super().__init__()
        
    #Correlation between numerical data
    def correlation(self):
        print (f'Correlation between numerical data\n {self.dataset.corr()}\n')
        
    def plotHist(self):
        super().plot()
    
    def encoder(self, column_name, exclude_col = False):
        merged_df = self.dataset_ready.merge(pd.get_dummies(self.dataset_ready[column_name], drop_first=False, prefix=column_name), left_index=True, right_index=True)
        if exclude_col:
            del merged_df[column_name] # Exclude the original column
        return merged_df
    
    
    
if __name__ == '__main__':
    
    #path = 'https://raw.githubusercontent.com/acollant/fairfiles/main/compas-scores-raw.csv'
    path = 'Data/Datasets/Input/compas-scores-raw.csv'
    oCompas = CompasData()
    oCompas.path = path
    oCompas.readData()
    oCompas.dataInfo()
    oCompas.plot()
    oCompas.plotHist()
    
    category_features = ['ScoreText']
    oCompas.removeNaN(category_features)
   
    category_features = ['Ethnic_Code_Text']
    oCompas.removeNoise(category_features, 'African-Am', 'African-American')
   
    oCompas.drop_(['Person_ID', 'AssessmentID', 'Case_ID', 'Scale_ID', 'ScaleSet_ID', 'IsCompleted', 'IsDeleted', 'MiddleName', 'DateOfBirth', 'Screening_Date', 'RecSupervisionLevel','LastName','FirstName'])
    oCompas.correlation()
    
    oCompas.dataset_ready = oCompas.dataset.copy()
    category_features = ['Agency_Text', 'Sex_Code_Text', 'Ethnic_Code_Text', 'ScaleSet', 'AssessmentReason', 'Language', 'LegalStatus', 'CustodyStatus', 'MaritalStatus', 'RecSupervisionLevelText', 'DisplayText', 'ScoreText', 'AssessmentType']
    for cat in category_features:
        oCompas.dataset_ready = oCompas.encoder(cat, exclude_col = True)
        
        
    print(oCompas.dataset_ready.columns)
    
    oCompas.savetoCVS(r'Data/Datasets/Output/CompasDataset.csv');
    #print(oCompas.dataInfo())

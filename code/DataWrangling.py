### Importing the general libraries we shall use in this notebook
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting library that use matplot in background
import matplotlib.pyplot as plt # to plot some parameters in seaborn


class DataWrangling():
   
    def __init__(self):
        pass
    
    def __setattr__(self, attrname, value):
        self.__dict__[attrname] = value
    
    def printPath(self):
        print(f' the data is {self.path}')
    
    def readData(self, index_col_ = True):
        if index_col_: 
            self.dataset = pd.read_csv(self.path)
        else:
            self.dataset = pd.read_csv(self.path, index_col = 0)
        
    def getData(self):
        return self.dataset
    
    def setData(self, dataset):
        self.dataset = dataset 
        
    def distribution(self):
        for cols in self.dataset.columns:
            if self.dataset[cols].dtypes == 'int64': 
                print(f'Column {cols}\n {self.dataset[cols].value_counts(bins=10)}\n')
    
    def plot(self):
        sns.set_context('talk', font_scale = .9)
        self.dataset.hist(bins = 50, figsize = (20,15)) 
        plt.show()
          
    def printProtected(self):
        print (self.protected)
        dataset = self.dataset
        print(dataset[self.protected].value_counts())
        
    #Correlation between numerical data
    def correlation(self):
        print (f'Correlation between numerical data\n {self.dataset.corr()}\n')
          
    def drop_(self,cols):
        self.dataset = self.dataset.drop(columns = cols, axis = 1)
        print(f'drop dataset columns! {cols}\n')
        #print(f'Print first 3 rows\n {self.dataset.head(3)}\n')
        
    # Apply the new distribution to the dataset
    def applylog10Norm(self,cols):
        self.dataset[cols] = np.log10(self.dataset[cols])
       
    def applyMinMaxNorm(self,col):
        min = self.dataset[col].min()
        max = self.dataset[col].max()
        self.dataset[col] = (self.dataset[col] - min) / (max - min)        
    
    def savetoCVS(self,filename):
        self.dataset_ready.to_csv (filename, index = False, header=  True)
              
    #Let us split age into categories
    def setAgeByGroup(self, interval, groups, col, newCol):
        self.dataset[newCol] = pd.cut(x = self.dataset[col], bins = interval, labels = groups)
    
    #Remove value simulating Null   
    def removeNoise(self, cols, noise, newVal):
        for c in cols:
            self.dataset[c] = self.dataset[c].replace(noise,newVal)
            
    def removeNaN(self, cols):
        for c in cols:
            self.dataset[c] = self.dataset[c].fillna('no_inf')
    
    def dataInfo(self):
        # How much data do we have?
        print (f'The number of rows {self.dataset.shape[0]} and the number of columns {self.dataset.shape[1]}\n')
        
        # Display dataset columns
        print(f'Dataset columns and types\n {self.dataset.dtypes}\n')
        
        # Null data?
        print(f'Is there any NULL Columns\n {self.dataset.info()}\n')
        
        # How many missing values
        print (f'Number of missing values\n {self.dataset.isnull().sum()}\n')
               
        # Display row having missing data: we can use isnull or isna
        print(f'Display row having missing data:\n {self.dataset[self.dataset.isna().any(axis=1)]}\n')
        
    
        # What are the types of features (numerical vs categorical) ?
        print(f'Summary of the numerical attribute\n {self.dataset.describe()}')


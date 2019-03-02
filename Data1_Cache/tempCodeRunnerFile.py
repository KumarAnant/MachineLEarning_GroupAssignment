import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir("E:/OneDrive/Hult/Machine Learning/Assignments/Group Assignment/Data")
file = 'Ames Housing Dataset.xls'

data = pd.read_excel("birthweight.xlsx")
print(data.head())

print(data.columns)


print(data.describe)

print(data.isnull().sum())

# Create flag for missing values
for col in data:
      
    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """

    if data[col].isnull().any():
        data['m_'+col] = data[col].isnull().astype(int)

print(data.head())
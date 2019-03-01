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

data_dropped = data.dropna()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['meduc'], kde=False)
plt.xlabel("Grade")
plt.ylabel("Counts")
plt.title("Mothers Education")
plt.show()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['monpre'], kde=False)
plt.xlabel("Months")
plt.ylabel("Counts")
plt.title("Mnth Pre Natal Care began")
plt.show()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['npvis'], kde=False)
plt.xlabel("Visits")
plt.ylabel("Counts")
plt.title("Total Number of Parental Visits")
plt.show()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['fage'], kde=False)
plt.xlabel("Age")
plt.ylabel("Counts")
plt.title("Fathers Age, years")
plt.show()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['feduc'], kde=False)
plt.xlabel("Education")
plt.ylabel("Counts")
plt.title("Fathers Education, Years")
plt.show()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['omaps'], kde= False)
plt.xlabel("Score")
plt.ylabel("Counts")
plt.title("One Minute Apgar Score")
plt.show()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['fmaps'], kde=False)
plt.xlabel("Score")
plt.ylabel("Counts")
plt.title("Five Minute Apgar Score")
plt.show()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['cigs'], kde=False)
plt.xlabel("Number")
plt.ylabel("Counts")
plt.title("Average Cigarettes per Day")
plt.show()

plt.figure(figsize=(20, 15), dpi=100)
sns.distplot(data_dropped['drink'], kde=False)
plt.xlabel("Numbers")
plt.ylabel("Counts")
plt.title("Average drinks per day")
plt.show()
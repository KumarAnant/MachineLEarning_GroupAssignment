#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:44:26 2019

@author: Martina Cerina

Workdirectory: /Users/ludovicaflocco/Desktop/Machine_Learning
"""

# Loading Libraries
import pandas as pd
import statsmodels.formula.api as smf # regression modeling
import seaborn as sns
import matplotlib.pyplot as plt
import os

### For KNN model
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation

os.chdir('E:/OneDrive/Hult/Machine Learning/Assignments/Group Assignment/Data2')
file = 'birthweight_feature_set.xlsx'

birthweight = pd.read_excel(file)

########################
# Fundamental Dataset Exploration
########################

# Column names
birthweight.columns
#mage = mother's age 
#meduc = mother's education
#monpre = month prenatal care began 
#npvis = number of prenatal visits
#fage = father's age
#feduc = father's education 
#omaps = one minute apgar score 
#fmaps = five minutes apgar score 
#cigs = average cigarettes per day
#drink = average drink per day 
#male = 1 if baby male 
#mwhte = 1 if mother white
#mblck = 1 if mothe black 
#moth = 1 if mothe is other 
#fwhte = 1 if father white 
#fblck = 1 if father black 
#goth = 1 if father others
#bwght = birthweigh grams 


# Dimensions of the DataFrame
birthweight.shape #18 variables

# Information about each variable 
birthweight.info()

# Descriptive statistics (IS IT RELEVANT NOW?)
birthweight.describe().round(2)

#NEED TO CHECK IT : DON'T THINK WE NEED IT
birthweight.sort_values('bwght', ascending = False)

###############################################################################
# Imputing Missing Values
###############################################################################

print(
      birthweight
      .isnull()
      .sum()
      )


for col in birthweight:#This is something we should always be trying to do

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if birthweight[col].isnull().any():
        birthweight['m_'+col] = birthweight[col].isnull().astype(int)
        
#if there is a null value in any column i want to flag it 
        
#meduc      30
#monpre      5
#npvis      68
#fage        6
#feduc      47
#omaps       3
#fmaps       3
#cigs      110
#drink     115        
        
df_dropped = birthweight.dropna()

#Creating histograms to check if normal distributed or not
sns.distplot(df_dropped['meduc']) #NOT NORMALLY DISTRIBUTED UNLESS DIVIDED INTO DIFF STEPS
sns.distplot(df_dropped['monpre']) #NOT NORMALLY DISTRIBUTED UNLESS DIVIDED INTO DIFF STEPS
sns.distplot(df_dropped['npvis']) #KINDA NORMALLY --> MEAN
sns.distplot(df_dropped['fage']) #YES NORMALLY --> MEAN
sns.distplot(df_dropped['feduc']) #NOT NORMALLY DISTRIBUED
sns.distplot(df_dropped['omaps']) #NOT NORMALLY DISTRIBUTED
sns.distplot(df_dropped['fmaps']) #TAKE THE MEAN 
sns.distplot(df_dropped['cigs']) #NOT
sns.distplot(df_dropped['drink']) #zero inflated

# # drink is zero inflated. Imputing with zero.
# fill = 0

#MEAN IMPUTATION FOR NPVIS VARIABLE

fill = birthweight['npvis'].median()

birthweight['npvis'] = birthweight['npvis'].fillna(fill)

fill = birthweight['meduc'].median()

birthweight['meduc'] = birthweight['meduc'].fillna(fill)


# fill = birthweight['monpre'].median()

# birthweight['monpre'] = birthweight['monpre'].fillna(fill)


fill = birthweight['feduc'].median()

birthweight['feduc'] = birthweight['feduc'].fillna(fill)



# Checking the overall dataset to verify that there are no missing values remaining
print(
      birthweight
      .isnull()
      .any()
      .any()
      )

###############################################################################
# Outlier Analysis
###############################################################################

birthweight_quantiles = birthweight.loc[:, :].quantile([0.20,
                                                0.40,
                                                0.60,
                                                0.80,
                                                1.00])

    
print(birthweight_quantiles)

for col in birthweight:
    print(col)

########################
# Visual EDA (Histograms)
########################


plt.subplot(2, 2, 1)
sns.distplot(birthweight['mage'],
             color = 'g')

plt.xlabel('mage')


sns.boxplot(x =birthweight['mage'])


########################


plt.subplot(2, 2, 2)
sns.distplot(birthweight['meduc'],
             color = 'y')

plt.xlabel('meduc')



########################


plt.subplot(2, 2, 3)
sns.distplot(birthweight['monpre'],
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('monpre')



########################


plt.subplot(2, 2, 4)

sns.distplot(birthweight['npvis'],
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('npvis')



plt.tight_layout()
# plt.savefig('Birthweight Histograms 1.png')


########################
########################



plt.subplot(2, 2, 1)
sns.distplot(birthweight['fage'],
             color = 'g')

plt.xlabel('fage')


########################

plt.subplot(2, 2, 2)
sns.distplot(birthweight['feduc'],
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('feduc')



########################


plt.subplot(2, 2, 3)

sns.distplot(birthweight['omaps'],
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('omaps')


########################

plt.subplot(2, 2, 4)
sns.distplot(birthweight['fmaps'],
             color = 'y')

plt.xlabel('fmaps')




plt.tight_layout()
# plt.savefig('Birthweight Data Histograms 2.png')

plt.show()


########################
########################


plt.subplot(2, 2, 1)
sns.distplot(birthweight['cigs'],
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('cigs')



########################

plt.subplot(2, 2, 2)

sns.distplot(birthweight['drink'],
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('drink')



########################

plt.subplot(2, 2, 3)
sns.distplot(birthweight['bwght'],
             color = 'g')

plt.xlabel('bwght')


plt.tight_layout()
# plt.savefig('Birthweight Data Histograms 3.png')


########################
# Tuning and Flagging Outliers
########################

birthweight_quantiles = birthweight.loc[:, :].quantile([0.05,
                                                0.40,
                                                0.60,
                                                0.80,
                                                0.95])

print(birthweight_quantiles['mage'])
"""

Assumed Continuous/Interval Variables - 

Mother's age
Mother's education
Month Prenatal Care began
Number of Prenatal Visits
Father's Age
Father's Education
One Minute apgar score
Five Minute's apgar score
Average cigaretts per day
Average drinks per day
Birthweigh grams 

Binary Classifiers -

Baby male 
Mother white
Mothe black 
Mother is other 
Father white 
Father black 
Father others

"""


# Outlier flags
mage_low = 20

mage_high = 55

overall_low_meduc = 10

monpre_low = 0

monpre_high = 7

npvis_low = 5

npvis_high = 18

fage_low = 20

fage_high = 62

overall_low_feduc = 7

overall_low_omaps = 4

overall_low_fmaps = 6

overall_cigs = 19

bwght_low = 2500

bwght_high = 4500

overall_drink = 11

########################
# Create a new column for Race and cEdu

birthweight['race'] = 0
birthweight['cEdu'] = 0
abc = birthweight['cigs']
for val in enumerate(birthweight.loc[ : , 'fwhte']):
      birthweight.loc[val[0], 'race'] =   str(birthweight.loc[val[0], 'mwhte']) + \
                                          str(birthweight.loc[val[0], 'mblck']) + \
                                          str(birthweight.loc[val[0], 'moth']) + \
                                          str(birthweight.loc[val[0], 'fwhte']) + \
                                          str(birthweight.loc[val[0], 'fblck']) + \
                                          str(birthweight.loc[val[0], 'foth'])
      birthweight.loc[val[0], 'cEdu'] =   birthweight.loc[val[0], 'meduc'] + \
                                          birthweight.loc[val[0], 'feduc']

########################
# Creating Outlier Flags
########################

# Building loops for outlier imputation

########################
# Mage

birthweight['out_mage'] = 0


for val in enumerate(birthweight.loc[ : , 'mage']):
    
    if val[1] >= mage_high:
        birthweight.loc[val[0], 'out_mage'] = 1
        
    if val[1] <= mage_low:
        birthweight.loc[val[0], 'out_mage'] = -1
        

########################
# Meduc

birthweight['out_meduc'] = 0


for val in enumerate(birthweight.loc[ : , 'meduc']):
            
    if val[1] <= overall_low_meduc:
        birthweight.loc[val[0], 'out_meduc'] = -1


########################
# Monpre

birthweight['out_monpre'] = 0


for val in enumerate(birthweight.loc[ : , 'monpre']):
    
    if val[1] >= monpre_high:
        birthweight.loc[val[0], 'out_monpre'] = 1
        
    if val[1] <= monpre_low:
        birthweight.loc[val[0], 'out_monpre'] = -1

########################
# Npvis

birthweight['out_npvis'] = 0

for val in enumerate(birthweight.loc[ : , 'npvis']):
    
    if val[1] >= npvis_high:
        birthweight.loc[val[0], 'out_npvis'] = 1
        
    if val[1] <= npvis_low:
        birthweight.loc[val[0], 'out_npvis'] = -1
        
        
########################
# Fage

birthweight['out_fage'] = 0

for val in enumerate(birthweight.loc[ : , 'fage']):
    
    if val[1] >= fage_high:
        birthweight.loc[val[0], 'out_fage'] = 1
        
    if val[1] <= fage_low:
        birthweight.loc[val[0], 'out_fage'] = -1
        
########################
# Feduc

birthweight['out_feduc'] = 0

for val in enumerate(birthweight.loc[ : , 'feduc']):   
        
    if val[1] <= overall_low_feduc:
        birthweight.loc[val[0], 'out_feduc'] = -1


########################
# Omaps

birthweight['out_omaps'] = 0

for val in enumerate(birthweight.loc[ : , 'omaps']):
        
    if val[1] <= overall_low_omaps:
        birthweight.loc[val[0], 'out_omaps'] = -1

########################
# Fmaps       

birthweight['out_fmaps'] = 0

for val in enumerate(birthweight.loc[ : , 'fmaps']):

    if val[1] <= overall_low_fmaps:
        birthweight.loc[val[0], 'out_fmaps'] = -1
        
########################
# Cigs

birthweight['out_cigs'] = 0

for val in enumerate(birthweight.loc[ : , 'cigs']):
            
    if val[1] >= overall_cigs:
        birthweight.loc[val[0], 'out_cigs'] = 1
        
########################
# Bwght

birthweight['out_bwght'] = 0

for val in enumerate(birthweight.loc[ : , 'bwght']):
    
    if val[1] >= bwght_high:
        birthweight.loc[val[0], 'out_bwght'] = 1
        
    if val[1] <= bwght_low:
        birthweight.loc[val[0], 'out_bwght'] = -1
        
########################
# Drink

birthweight['out_drink'] = 0

for val in enumerate(birthweight.loc[ : , 'drink']):
            
    if val[1] >= overall_drink:
        birthweight.loc[val[0], 'out_drink'] = 1



###############################################################################
# Correlation Analysis
###############################################################################

birthweight.head()


df_corr = birthweight.corr().round(2)


print(df_corr)


df_corr.loc['bwght'].sort_values(ascending = False)
        
########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))
 
df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


# plt.savefig('Variable Correlation Heatmap.png')
plt.show()

# birthweight.to_excel('Birthweight_explored.xlsx')

#help(corr)
#help(pd.corr)
#help(pd.DataFrame.corr)
#We are often dealing with pearson correlation 
#with unsupervised we will experience Spearman correlation   

########################
# SUMMARY BEFORE RUNNING THE MODEL
########################
      
#print(birthweight.head())

fmaps_dummies = pd.get_dummies(list(birthweight['fmaps']), prefix = 'fmaps', drop_first = True)
omaps_dummies = pd.get_dummies(list(birthweight['omaps']), prefix = 'omaps', drop_first = True)
drink_dummies = pd.get_dummies(list(birthweight['drink']), prefix = 'drink', drop_first = True)
meduc_dummies = pd.get_dummies(list(birthweight['meduc']), prefix = 'meduc', drop_first = True)
feduc_dummies = pd.get_dummies(list(birthweight['feduc']), prefix = 'feduc', drop_first = True)
race_dummies = pd.get_dummies(list(birthweight['race']), prefix = 'race', drop_first = True)
npvis_dummies = pd.get_dummies(list(birthweight['npvis']), prefix = 'npvis', drop_first = True)
cigs_dummies = pd.get_dummies(list(birthweight['cigs']), prefix = 'cigs', drop_first = True)
# male_dummies = pd.get_dummies(list(birthweight['male']), prefix = 'male', drop_first = True)
# mwhte_dummies = pd.get_dummies(list(birthweight['mwhte']), prefix = 'mwhte', drop_first = True)
# mblck_dummies = pd.get_dummies(list(birthweight['mblck']), prefix = 'mblck', drop_first = True)
# moth_dummies = pd.get_dummies(list(birthweight['moth']), prefix = 'moth', drop_first = True)
# fwhte_dummies = pd.get_dummies(list(birthweight['fwhte']), prefix = 'fwhte', drop_first = True)
# fblck_dummies = pd.get_dummies(list(birthweight['fblck']), prefix = 'fblck', drop_first = True)
# foth_dummies = pd.get_dummies(list(birthweight['foth']), prefix = 'foth', drop_first = True)

birthweight_2 = pd.concat(
        [birthweight.loc[:,:],
         fmaps_dummies, drink_dummies, 
         meduc_dummies, race_dummies,
         omaps_dummies, npvis_dummies,
         cigs_dummies, feduc_dummies],
         axis = 1)

birthweight_2['C_wM'] = birthweight_2['male'] * birthweight_2['mwhte']
birthweight_2['C_bM'] = birthweight_2['male'] * birthweight_2['mblck']
birthweight_2['C_oM'] = birthweight_2['male'] * birthweight_2['moth']
birthweight_2['C_bF'] = birthweight_2['male'] * birthweight_2['fblck']
birthweight_2['C_bM'] = birthweight_2['male'] * birthweight_2['mblck']
birthweight_2['cigolic'] = birthweight_2['cigs'] * birthweight_2['drink']
birthweight_2['edu'] = birthweight_2['feduc'] * birthweight_2['meduc']
birthweight_2['oth'] = birthweight_2['foth'] * birthweight_2['moth']
# birthweight_2['blck'] = birthweight_2['fblck'] * birthweight_2['mblck']  DOes not give result
# birthweight_2['whte'] = birthweight_2['fwhte'] * birthweight_2['mwhte']


#birthweight_2.to_excel('abcd.xlsx')
lm_full = smf.ols(formula = """bwght ~    mage +                                          
                                          monpre +                                          
                                          fage +
                                          birthweight_2['feduc_7.0'] +
                                          birthweight_2['feduc_8.0'] +
                                          birthweight_2['feduc_10.0'] +
                                          birthweight_2['feduc_11.0'] +
                                          birthweight_2['feduc_12.0'] +
                                          birthweight_2['feduc_13.0'] +
                                          birthweight_2['feduc_14.0'] +
                                          birthweight_2['feduc_15.0'] +
                                          birthweight_2['feduc_16.0'] +
                                          birthweight_2['feduc_17.0'] +                                         
                                          birthweight_2['cigs_1'] +
                                          birthweight_2['cigs_2'] +
                                          birthweight_2['cigs_3'] +
                                          birthweight_2['cigs_4'] +
                                          birthweight_2['cigs_5'] +
                                          birthweight_2['cigs_6'] +
                                          birthweight_2['cigs_7'] +
                                          birthweight_2['cigs_8'] +
                                          birthweight_2['cigs_9'] +
                                          birthweight_2['cigs_10'] +
                                          birthweight_2['cigs_11'] +
                                          birthweight_2['cigs_12'] +
                                          birthweight_2['cigs_13'] +
                                          birthweight_2['cigs_14'] +
                                          birthweight_2['cigs_15'] +
                                          birthweight_2['cigs_16'] +
                                          birthweight_2['cigs_17'] +
                                          birthweight_2['cigs_18'] +
                                          birthweight_2['cigs_19'] +
                                          birthweight_2['cigs_20'] +
                                          birthweight_2['cigs_21'] +
                                          birthweight_2['cigs_22'] +
                                          birthweight_2['cigs_23'] +
                                          birthweight_2['cigs_24'] +
                                          birthweight_2['cigs_25'] +                                          
                                          male +
                                          mwhte +
                                          mblck +
                                          moth +
                                          fwhte +
                                          fblck +
                                          foth +                                          
                                          m_meduc +
                                          m_npvis +
                                          m_feduc +                                          
                                          cEdu +
                                          out_mage +
                                          out_meduc +
                                          out_monpre +
                                          out_npvis +
                                          out_fage +
                                          out_feduc +
                                          out_omaps +
                                          out_fmaps +
                                          out_cigs +
                                          out_bwght +
                                          out_drink +
                                          birthweight_2['omaps_3'] +
                                          birthweight_2['omaps_4'] +
                                          birthweight_2['omaps_5'] +
                                          birthweight_2['omaps_6'] +
                                          birthweight_2['omaps_7'] +
                                          birthweight_2['omaps_8'] +
                                          birthweight_2['omaps_9'] +
                                          birthweight_2['omaps_10'] +                                          
                                          birthweight_2['fmaps_6'] +
                                          birthweight_2['fmaps_7'] +
                                          birthweight_2['fmaps_8'] +
                                          birthweight_2['fmaps_9'] +
                                          birthweight_2['fmaps_10'] +
                                          birthweight_2['drink_1'] +
                                          birthweight_2['drink_2'] +
                                          birthweight_2['drink_3'] +
                                          birthweight_2['drink_4'] +
                                          birthweight_2['drink_5'] +
                                          birthweight_2['drink_6'] +
                                          birthweight_2['drink_7'] +
                                          birthweight_2['drink_8'] +
                                          birthweight_2['drink_9'] +
                                          birthweight_2['drink_10'] +
                                          birthweight_2['drink_11'] +
                                          birthweight_2['drink_12'] +
                                          birthweight_2['drink_13'] +
                                          birthweight_2['drink_14'] +
                                          birthweight_2['meduc_10.0'] +
                                          birthweight_2['meduc_11.0'] +
                                          birthweight_2['meduc_12.0'] +
                                          birthweight_2['meduc_13.0'] +
                                          birthweight_2['meduc_14.0'] +
                                          birthweight_2['meduc_15.0'] +
                                          birthweight_2['meduc_16.0'] +
                                          birthweight_2['meduc_17.0'] +
                                          birthweight_2['race_001010'] + 
                                          birthweight_2['race_001100'] +
                                          birthweight_2['race_010001'] +
                                          birthweight_2['race_010010'] +
                                          birthweight_2['race_010100'] +
                                          birthweight_2['race_100100'] +
                                          birthweight_2['npvis_3.0'] +
                                          birthweight_2['npvis_5.0'] +
                                          birthweight_2['npvis_6.0'] +
                                          birthweight_2['npvis_7.0'] +
                                          birthweight_2['npvis_8.0'] +
                                          birthweight_2['npvis_9.0'] +
                                          birthweight_2['npvis_10.0'] +
                                          birthweight_2['npvis_11.0'] +
                                          birthweight_2['npvis_12.0'] +
                                          birthweight_2['npvis_13.0'] +
                                          birthweight_2['npvis_14.0'] +
                                          birthweight_2['npvis_15.0'] +
                                          birthweight_2['npvis_16.0'] +
                                          birthweight_2['npvis_17.0'] +
                                          birthweight_2['npvis_18.0'] +
                                          birthweight_2['npvis_19.0'] +
                                          birthweight_2['npvis_20.0'] +
                                          birthweight_2['npvis_25.0'] +
                                          birthweight_2['npvis_30.0'] +
                                          birthweight_2['npvis_31.0'] +
                                          birthweight_2['npvis_35.0']
                                          """,
                  data = birthweight_2)


# Fitting Results
results = lm_full.fit()



# Printing Summary Statistics
results.pvalues

print(results.summary())

rsq_lm_full = results.rsquared.round(3)

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
    

########################
# Significant Model
########################

lm_significant = smf.ols(formula = """bwght ~   mage +                                          
                                          monpre +                                          
                                          fage +
                                          feduc +                                          
                                          cigs +
                                          drink +
                                          male +                                        
                                          m_meduc +
                                          m_npvis +
                                          m_feduc +
                                          race +
                                          cEdu +
                                          out_mage +
                                          out_meduc +
                                          out_monpre +
                                          out_npvis +
                                          out_fage +
                                          out_feduc +
                                          out_omaps +
                                          out_fmaps +
                                          out_cigs +
                                          out_bwght +
                                          out_drink +
                                          birthweight_2['omaps_3'] +
                                          birthweight_2['omaps_4'] +
                                          birthweight_2['omaps_5'] +
                                          birthweight_2['omaps_6'] +
                                          birthweight_2['omaps_7'] +
                                          birthweight_2['omaps_8'] +
                                          birthweight_2['omaps_9'] +
                                          birthweight_2['omaps_10'] +                                          
                                          birthweight_2['fmaps_6'] +
                                          birthweight_2['fmaps_7'] +
                                          birthweight_2['fmaps_8'] +
                                          birthweight_2['fmaps_9'] +
                                          birthweight_2['fmaps_10'] +
                                          birthweight_2['meduc_10.0'] +
                                          birthweight_2['meduc_11.0'] +
                                          birthweight_2['meduc_12.0'] +
                                          birthweight_2['meduc_13.0'] +
                                          birthweight_2['meduc_14.0'] +
                                          birthweight_2['meduc_15.0'] +
                                          birthweight_2['meduc_16.0'] +
                                          birthweight_2['meduc_17.0'] +
                                          birthweight_2['race_001010'] + 
                                          birthweight_2['race_001100'] +
                                          birthweight_2['race_010001'] +
                                          birthweight_2['race_010010'] +
                                          birthweight_2['race_010100'] +
                                          birthweight_2['race_100100'] +
                                          birthweight_2['npvis_3.0'] +
                                          birthweight_2['npvis_5.0'] +
                                          birthweight_2['npvis_6.0'] +
                                          birthweight_2['npvis_7.0'] +
                                          birthweight_2['npvis_8.0'] +
                                          birthweight_2['npvis_9.0'] +
                                          birthweight_2['npvis_10.0'] +
                                          birthweight_2['npvis_11.0'] +
                                          birthweight_2['npvis_12.0'] +
                                          birthweight_2['npvis_13.0'] +
                                          birthweight_2['npvis_14.0'] +
                                          birthweight_2['npvis_15.0'] +
                                          birthweight_2['npvis_16.0'] +
                                          birthweight_2['npvis_17.0'] +
                                          birthweight_2['npvis_18.0'] +
                                          birthweight_2['npvis_19.0'] +
                                          birthweight_2['npvis_20.0'] +
                                          birthweight_2['npvis_25.0'] +
                                          birthweight_2['npvis_30.0'] +
                                          birthweight_2['npvis_31.0'] +
                                          birthweight_2['npvis_35.0']
                                          """,                                                
                  data = birthweight_2)

# Fitting Results
results = lm_significant.fit()
results.rsquared_adj.round(3)


# Printing Summary Statistics
print(results.summary())

rsq_lm_significant = results.rsquared.round(3)

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")


#########################################
######### KNN FUll Model ################
#########################################

birthweight_2_data = birthweight_2.drop(['bwght'], axis = 1)

birthweight_2_target = birthweight_2.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(
            birthweight_2_data,
            birthweight_2_target,
            test_size = 0.1,
            random_state = 508)

# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)

# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
birthweight_2_train = pd.concat([X_train, y_train], axis = 1)
birthweight_2_test = pd.concat([X_test, y_test], axis = 1)



# ########################
# # Step 1: Create a model object
# ########################

# # Creating a regressor object
# knn_reg = KNeighborsRegressor(algorithm = 'auto',
#                               n_neighbors = 1)



# # Checking the type of this new object
# type(knn_reg)


# # Teaching (fitting) the algorithm based on the training data
# knn_reg.fit(X_train, y_train)

# # Predicting on the X_data that the model has never seen before
# y_pred = knn_reg.predict(X_test)



# # Printing out prediction values for each test observation
# print(f"""
# Test set predictions:
# {y_pred}
# """)



# # Calling the score method, which compares the predicted values to the actual
# # values
# y_score = knn_reg.score(X_test, y_test)



# # The score is directly comparable to R-Square
# print(y_score)



#########################################
########### Using Stats MethodL #########
#########################################

lm_significant_stats = smf.ols(formula = """bwght ~   mage +
                                          meduc +
                                          monpre +                                          
                                          fage +
                                          feduc +                                          
                                          cigs +
                                          drink +
                                          male +                                        
                                          m_meduc +
                                          m_npvis +
                                          m_feduc +
                                          race +
                                          cEdu +
                                          out_mage +
                                          out_meduc +
                                          out_monpre +
                                          out_npvis +
                                          out_fage +
                                          out_feduc +
                                          out_omaps +
                                          out_fmaps +
                                          out_cigs +
                                          out_bwght +
                                          out_drink +
                                          birthweight_2_train['omaps_3'] +
                                          birthweight_2_train['omaps_4'] +
                                          birthweight_2_train['omaps_5'] +
                                          birthweight_2_train['omaps_6'] +
                                          birthweight_2_train['omaps_7'] +
                                          birthweight_2_train['omaps_8'] +
                                          birthweight_2_train['omaps_9'] +
                                          birthweight_2_train['omaps_10'] +                                          
                                          birthweight_2_train['fmaps_6'] +
                                          birthweight_2_train['fmaps_7'] +
                                          birthweight_2_train['fmaps_8'] +
                                          birthweight_2_train['fmaps_9'] +
                                          birthweight_2_train['fmaps_10'] +
                                          birthweight_2_train['meduc_10.0'] +
                                          birthweight_2_train['meduc_11.0'] +
                                          birthweight_2_train['meduc_12.0'] +
                                          birthweight_2_train['meduc_13.0'] +
                                          birthweight_2_train['meduc_14.0'] +
                                          birthweight_2_train['meduc_15.0'] +
                                          birthweight_2_train['meduc_16.0'] +
                                          birthweight_2_train['meduc_17.0'] +
                                          birthweight_2_train['race_001010'] + 
                                          birthweight_2_train['race_001100'] +
                                          birthweight_2_train['race_010001'] +
                                          birthweight_2_train['race_010010'] +
                                          birthweight_2_train['race_010100'] +
                                          birthweight_2_train['race_100100'] +
                                          birthweight_2_train['npvis_3.0'] +
                                          birthweight_2_train['npvis_5.0'] +
                                          birthweight_2_train['npvis_6.0'] +
                                          birthweight_2_train['npvis_7.0'] +
                                          birthweight_2_train['npvis_8.0'] +
                                          birthweight_2_train['npvis_9.0'] +
                                          birthweight_2_train['npvis_10.0'] +
                                          birthweight_2_train['npvis_11.0'] +
                                          birthweight_2_train['npvis_12.0'] +
                                          birthweight_2_train['npvis_13.0'] +
                                          birthweight_2_train['npvis_14.0'] +
                                          birthweight_2_train['npvis_15.0'] +
                                          birthweight_2_train['npvis_16.0'] +
                                          birthweight_2_train['npvis_17.0'] +
                                          birthweight_2_train['npvis_18.0'] +
                                          birthweight_2_train['npvis_19.0'] +
                                          birthweight_2_train['npvis_20.0'] +
                                          birthweight_2_train['npvis_25.0'] +
                                          birthweight_2_train['npvis_30.0'] +
                                          birthweight_2_train['npvis_31.0'] +
                                          birthweight_2_train['npvis_35.0']
                                          """,
                  data = birthweight_2_train)
# Fitting Results
results = lm_significant.fit()



# Printing Summary Statistics
print(results.summary())

rsq_lm_stat_significant = results.rsquared.round(3)

#rsq_lm_stat_significant_test = lm_significant_stats.predict(X_test)

###############################################################################
# Applying the Optimal Model in scikit-learn
###############################################################################


birthweight_data   = birthweight_2.loc[:,['mage',                                         
                                          'cigs',                                 
                                          'male',
                                          'mwhte',
                                          'mblck',
                                          'moth',
                                          'fwhte',
                                          'fblck',
                                          'foth',
                                          'm_meduc',
                                          'm_npvis',
                                          'm_feduc',
                                          'cEdu',
                                          'C_wM',
                                          'C_bM',
                                          'C_bF',
                                          'oth',
                                          'cigolic',
                                          'edu',
                                          'out_mage',
                                          'out_meduc',
                                          'out_monpre',
                                          'out_npvis',
                                          'out_fage',
                                          'out_feduc',                                          
                                          'out_cigs',
                                          'out_bwght',
                                          'out_drink',
                                          'drink_4',
                                          'drink_5',
                                          'drink_6',
                                          'drink_7',
                                          'drink_8',
                                          'drink_9',
                                          'drink_10',
                                          'drink_11',
                                          'drink_12',
                                          'drink_13',
                                          'drink_14',
                                          'npvis'                                                                                                                  
                                          ]]

# Preparing the target variable
birthweight_target = birthweight_2.loc[:, 'bwght']


# Same code as before
X_train, X_test, y_train, y_test = train_test_split(
            birthweight_data,
            birthweight_target,
            test_size = 0.1,
            random_state = 508)

########################
# Using KNN  on the optimal model (same code as before)
########################

# Exact loop as before
training_accuracy = []
test_accuracy = []



neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

print(max(test_accuracy))

print("Best test accuracy at N = ",test_accuracy.index(max(test_accuracy)))

########################
# The best results occur when k = 3.
########################

# Building a model with k = 3
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = test_accuracy.index(max(test_accuracy)))



# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)



# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score_knn_optimal)



# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

# Predictions
y_pred = knn_reg.predict(X_test)
print(f"""
Test set predictions:
{y_pred.round(2)}
""")

########################
## Does OLS predict better than KNN?
########################

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
            birthweight_data,
            birthweight_target,
            test_size = 0.1,
            random_state = 508)



# Prepping the Model
lr = LinearRegression(fit_intercept = False)

# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{lr_pred.round(2)}
""")

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)


# The score is directly comparable to R-Square
print("Fit score of scikit LR model: ",y_score_ols_optimal)


# Let's compare the testing score to the training score.

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

cv_lr_3 = cross_val_score(lr,
                          birthweight_data,
                          birthweight_target,
                          cv = 3)

print("Cross validation score of LR: ", (pd.np.mean(cv_lr_3)))

"""
Prof. Chase:
    These values are much lower than what we saw before when we didn't create
    a train/test split. However, these results are realistic given we have
    a better understanding as to how well our model will predict on new data.
"""


# Printing model results
print(f"""
Optimal model KNN score:      {y_score_knn_optimal.round(3)}
Optimal model OLS score:      {y_score_ols_optimal.round(3)}
CrossValidation (CV 3) score: {pd.np.mean(cv_lr_3).round(3)}
R-Square LM Full:             {rsq_lm_full.round(3)}
R-Square LM Signf:            {rsq_lm_significant.round(3)}
R-Square LM Signf (stat):     {rsq_lm_stat_significant.round(3)}
""")
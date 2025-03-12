import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')
# print(df.info())
df = df.rename(columns ={'Invoice ID':'Invoice_ID','Customer type' : 'Customer_type','Product line':'Product_line','Unit price':'Unit_price','Tax 5%':'Tax_5','gross margin percentage':'gross_margin','gross income' :'gross_income'})

# print(df.duplicated().sum())
# print(df.isna().sum())

cats = ['Invoice_ID', 'Branch', 'City', 'Customer_type', 'Gender', 'Product_line', 'Date', 'Time', 'Payment']
nums = ['Unit_price', 'Quantity', 'Tax_5%', 'Total', 'cogs', 'gross_margin', 'gross_income', 'Rating']

# print('categorical columns----->',cats)
# print('\t')
# print('\t')
# print('numerical columns------->',nums)

# print(df['Date'].unique())
df['Date'] = pd.to_datetime(df['Date'],errors='coerce')
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df = df.drop('Date',axis=1)

# print(df['Invoice_ID'].unique())
df['Invoice_ID'] = df['Invoice_ID'].str.replace("-",'')
df['Invoice_ID'] = df['Invoice_ID'].astype('int')

df['Time'] = df['Time'].str.replace(':','')
df['Time'] = df['Time'].astype('int')

from sklearn.preprocessing import LabelEncoder
encode  = LabelEncoder()

for i in df.columns:
    if df[i].dtype == 'object':
        df[i] = encode.fit_transform(df[i])
        

x = df.drop('Quantity',axis=1)
y = df['Quantity']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# # print(x_train.shape)
# # print(x_test.shape)
# # print(y_train.shape)
# # print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,r2_score
logistic = LogisticRegression()
logistic.fit(x_train,y_train)
y_pred = logistic.predict(x_test)

# # print("Mean Square Error",mean_squared_error(y_test,y_pred))  #Mean Square Error 7.39
# # print("R2 Score",r2_score(y_test,y_pred)) #R2 Score 0.15542857142857147

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
model = LogisticRegression()

params = {
    'penalty' : ['l1','l2','elastinet'],
    'C' :[0.1,0.01,1.0],
    'solver': ['lbfgs','liblinear','newton-cholesky']
}

tuning = GridSearchCV(model,param_grid=params,cv =5)
tuning.fit(x_train,y_train)
# print(tuning.best_params_) #{'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}
# print(tuning.best_score_)  #0.5225

import pickle
pickle.dump(logistic,open("logistic.pkl","wb"))






#!/usr/bin/env python
# coding: utf-8

# # Welcome to my (attempted) Start-Up Viability Predictor

# 1) Import all necessary libraries

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# 2) Import both necessary data sets and label them appropriately

# a) first, import and format the start-up dataset
# 
# This is a dataset of almost 50k startups and their funding processes and whether or not they currently exist today. This is the data that I will train my model on.

# In[2]:


investments_url = "investments_VC.csv"
df = pd.read_csv(investments_url, encoding ="latin1")

#fix column names
df['market'] = df[' market ']
df['funding_total_usd'] = df[' funding_total_usd ']

df = df.drop(['name', 'homepage_url', 'permalink', 'state_code',  ' market ',' funding_total_usd ', 'category_list'], axis=1)
df = df.drop(['round_A','round_B','round_C','round_D','round_E' ,'round_F','round_G' ,'round_H'], axis=1)
df = df.drop(['debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity', 'post_ipo_debt', 'secondary_market', 'product_crowdfunding'], axis=1)
df = df.drop(['seed', 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note', 'founded_month', 'founded_quarter'], axis=1)

df=df.dropna(subset=['status','funding_total_usd','founded_at'])
df = df.drop(df[df.funding_total_usd == '-'].index)

df['fund1']=df['first_funding_at'].str.slice(0,7)
df['fundLast']=df['last_funding_at'].str.slice(0,7)
df['found']=df['founded_at'].str.slice(0,7)
df = df.drop(['first_funding_at','last_funding_at','founded_at' ,'founded_year', 'region'], axis=1)

df.head(5)


# b) then, import and format the consumer price index information
# 
# This information will be cross-refernced with the important "life-cycle" dates of the start-ups. This model will try to analyze the start-ups by the CPI when they were founded, when they raised their first round of funding, and when they raised their last round of funding

# In[3]:


CPI_url = 'https://pkgstore.datahub.io/core/cpi-us/cpiai_csv/data/b17bfacbda3c08e51cd13fe544b8fca4/cpiai_csv.csv'
df2 = pd.read_csv(CPI_url)


# #### Cross reference the notable dates with the CPI on those dates

# In[4]:


df2['month_year'] = pd.to_datetime(df2['Date']).dt.to_period('M')
df2['fund1']=df2['month_year'].astype(str)
df2['Index_firstFunding'] = df2['Index']
df2 = df2.drop(['Date', 'Inflation', 'Index','month_year'], axis=1)


# In[5]:


df3 = pd.read_csv(CPI_url)
df3['month_year'] = pd.to_datetime(df3['Date']).dt.to_period('M')
df3['fundLast']=df3['month_year'].astype(str)
df3['Index_lastFunding'] = df3['Index']
df3 = df3.drop(['Date', 'Inflation', 'Index','month_year'], axis=1)


# In[6]:


df4 = pd.read_csv(CPI_url)
df4['month_year'] = pd.to_datetime(df4['Date']).dt.to_period('M')
df4['found']=df4['month_year'].astype(str)
df4['Index_found'] = df4['Index']
df4 = df4.drop(['Date', 'Inflation', 'Index','month_year'], axis=1)


# ### Merge everything onto one dataframe (and print the top so we can see how nicely it came out!)

# In[7]:


df = pd.merge(df, df2, how='left', on = 'fund1')
df=df.dropna(subset=['Index_firstFunding'])

df = pd.merge(df, df3, how='left', on = 'fundLast')
df=df.dropna(subset=['Index_lastFunding'])

df = pd.merge(df, df4, how='left', on = 'found')
df=df.dropna(subset=['Index_found'])

df["label"] = df["status"]
df = df.drop(['status', 'fund1', 'fundLast', 'funding_total_usd', 'found'], axis=1)

df.head(10)


#     Drop all rows that have NaN in the "label" column because these data points won't be useful to us

# In[8]:


dfRegression=df.dropna(subset=['label'])


#     Use OneHotEncoder to convert the "market" and "country_code" columns to usable data.
#     These columns (hopefully) contain useful data that we want to extract for our model to use.
#     OneHotEncoder makes the data that comes as strings useful for our model that only accepts numerical data.

# In[9]:


from sklearn.preprocessing import OneHotEncoder

Encode = OneHotEncoder(handle_unknown = 'ignore').fit(dfRegression[['market', 'country_code']])

TrainEncode = pd.DataFrame(Encode.transform(dfRegression[['market', 'country_code']]).toarray())

dfRegression = dfRegression.join(TrainEncode)

dfRegression = dfRegression.drop(['city', 'market', 'country_code'], axis=1)


# In[10]:


dfRegression = dfRegression.dropna()


# The "Data" is everthing in the DataFrame minus the labels! This will become the "X" we will train and test with!

# In[11]:


dfRegressionData = dfRegression.drop(['label'], axis=1)

X = dfRegressionData


# Import NLTK: (for the record - this took me over 3 hours of troubleshooting)

# In[12]:


get_ipython().system('pip3 install nltk')
import nltk
nltk.download('punkt')


# Tokenize the "labels" then binarize the "labels". This will become the "y" that we train and test with!

# In[13]:


from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

dfRegression["label"] = dfRegression["label"].apply(word_tokenize)
y = dfRegression["label"]

mlb = MultiLabelBinarizer()
y = pd.DataFrame(mlb.fit_transform(y))


# In[14]:


print(mlb.classes_)


# Important to split into training and testing sets!

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# #### Pre-Processing Pipeline (say that 5 times fast)
#     1) Scale Values
#     2) Run Principal Component Analysis (to reduce dimensionality)
#     3) Run Random Forest Regressor model
#     
# It is good to have this in a pipeline because whatever we do to the training set, we must do to the testing set!
# This pipeline helps to keep us consistent!
# 
# 

# In[16]:


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=.95)),
    ('model', RandomForestRegressor(n_estimators=3))
])


#     How did I pick Random Forest you might ask?
#     Great question!
#     
#     Random Forest is an example of a "bagging" algorithm
#     This is because the different trees run in parallel to each other 
#     This type of algorithm is a good algorithm for datasets that are imbalanced
#     
#     This is relevant here becuase there are way more examples of "operating" start-ups than "closed" start-ups in 
#     the train/test data set 
#     This obviously doesn't reflect how start-ups behave in the real world, so this bagging algorithm is my attempt 
#     to get rid of this imbalance
#     
#     I made a LARGE number of estimators (i.e. trees) in order to prevent overfitting which was a big issue for me

# ### fit the pipeline (and the model!) :
# 
# #### sorry it takes so long to train! : /
# 
# According to my estimates it takes around 40 minutes to train. Maybe make a cup of coffee or two while you wait.

# In[17]:


fitted = pipe.fit(X_train, y_train)


# ## Let's check how the model did!

# In[18]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

predictionsTrain = fitted.predict(X_train).round()
predictions = fitted.predict(X_test).round()

print("Train Report", classification_report(y_train, predictionsTrain))
print("Test Report", classification_report(y_test, predictions))


# This is a confusion matrix of how accurate the model does when we predict on the data we trained on.

# In[19]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

fig, ax = plt.subplots(figsize=(3, 3))
ConfusionMatrixDisplay(confusion_matrix(y_train.values.argmax(axis=1), predictionsTrain.argmax(axis=1))).plot(ax=ax)


# This is a confusion matrix of how accurate the model does when we predict on new test data. If this is very different than the last confusion matrix it signifies that the model likely overfit (and we probably need more trees)

# In[20]:


fig, ax = plt.subplots(figsize=(3, 3))
ConfusionMatrixDisplay(confusion_matrix(y_test.values.argmax(axis=1), predictions.argmax(axis=1))).plot(ax=ax)


# #### This model is clearly far from perfect. If I had more time I would implement more ways to get rid of the inherent imbalance in the dataset. 

# ## Lets Predict With This Model!

# ### example prediction (you can fill in your own data!)
# #### IMPORTANT!! The CPI Data only goes until the end of 2013 so putting in a later date will result in an error
# ##### obviously this is NOT ideal and I hope to find better CPI data before I sell this model for a billion dollars to a big VC firm in the near future! :) 

# In[21]:


Market = 'Real Estate'
Country_code = 'USA'
Funding_rounds = '2'
fund1 = '2009-07'
fundLast = '2010-06'
found = '2009-01'


# #### You can modify the data points above but please don't modify anything below this!
# 
# Create a DataFrame with the given data:

# In[22]:


data = {"market": [Market], "country_code": [Country_code], "Rounds": [Funding_rounds], 
        "fund1": [fund1], "fundLast": [fundLast], "found":[found]}

predicting = pd.DataFrame(data)

print(predicting)


# Must do the same data preprocessing as with the train/test dataset (this means corss reference given dates with the CPI index on those dates!)

# In[23]:


df2 = pd.read_csv(CPI_url)
df2['month_year'] = pd.to_datetime(df2['Date']).dt.to_period('M')
df2['fund1']=df2['month_year'].astype(str)
df2['Index_firstFunding'] = df2['Index']
df2 = df2.drop(['Date', 'Inflation', 'Index','month_year'], axis=1)

df3 = pd.read_csv(CPI_url)
df3['month_year'] = pd.to_datetime(df3['Date']).dt.to_period('M')
df3['fundLast']=df3['month_year'].astype(str)
df3['Index_lastFunding'] = df3['Index']
df3 = df3.drop(['Date', 'Inflation', 'Index','month_year'], axis=1)

df4 = pd.read_csv(CPI_url)
df4['month_year'] = pd.to_datetime(df4['Date']).dt.to_period('M')
df4['found']=df4['month_year'].astype(str)
df4['Index_found'] = df4['Index']
df4 = df4.drop(['Date', 'Inflation', 'Index','month_year'], axis=1)

predicting = pd.merge(predicting, df2, how='left', on = 'fund1')

predicting = pd.merge(predicting, df3, how='left', on = 'fundLast')

predicting = pd.merge(predicting, df4, how='left', on = 'found')

predicting = predicting.drop(['fund1', 'fundLast', 'found'], axis=1)


# Run the SAME OneHotEncoding on this data to make the string data into usable numerical data (again, need to be consistent with the training data and the predicting data!)

# In[24]:


from sklearn.preprocessing import OneHotEncoder

predict = pd.DataFrame(Encode.transform(predicting[['market', 'country_code']]).toarray())

predicting = predicting.drop(['market', 'country_code'], axis=1)

prediction = predicting.join(predict)


# Scale, reduce dimensionality, and then predict on the same model as before.
# That pipeline sure comes in handy!!

# In[25]:


prediction = fitted.predict(prediction)


# Drum roll for the final prediction about the startup viability!

# In[26]:


print(prediction)


# Thanks for reading! This project was very difficult but ultimately super rewarding! It was cool to see my idea work (somewhat) and to watch everything we learned this semester come together! Thank you!!

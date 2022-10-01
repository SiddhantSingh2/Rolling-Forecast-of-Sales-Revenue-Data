#!/usr/bin/env python
# coding: utf-8

# ## Loading The Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing The Datasets

# In[2]:


df_train = pd.read_csv("train.csv", low_memory = False)
df_test = pd.read_csv("test.csv", low_memory = False)
df_store = pd.read_csv("store.csv", low_memory = False)


# ### Assessing the 'train.csv' dataset

# In[3]:


df_train.head(10)


# In[4]:


df_train.shape


# In[5]:


df_train.info()


# There are 1017209 rows with 9 features in the "train.csv" dataset, where 2 features are categorical (Date & StateHoliday) while the rest are numerical and contains the following fields:

# - Store: a unique Id for each store
# - Sales: the turnover for any given day (target variable).
# - Customers: the number of customers on a given day.
# - Open: an indicator for whether the store was open: 0 = closed, 1 = open.
# - Promo: indicates whether a store is running a promo on that day.
# - StateHoliday: indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. All schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
# - SchoolHoliday: indicates if the (Store, Date) was affected by the closure of public schools.

# ### Checking for missing values

# In[6]:


sns.heatmap(df_train.isnull())


# In[7]:


df_train.isna().sum()


# ### Checking for duplicates

# In[8]:


df_train.duplicated(subset=['Store', 'Date']).value_counts()


# ### Assessing the 'store.csv' dataset

# In[15]:


df_store.head(10)


# In[16]:


df_store.shape


# In[17]:


df_store.info()


# There are 1115 rows with 10 features in the "store.csv" dataset, where 3 features are categorical (StoreType, Assortment & PromoInterval) while the rest are numerical and contains the following fields:

# - Store: a unique Id for each store
# - StoreType: differentiates between 4 different store models: a, b, c, d
# - Assortment: describes an assortment level: a = basic, b = extra, c = extended
# - CompetitionDistance: distance in meters to the nearest competitor store
# - CompetitionOpenSince[Month/Year]: gives the approximate year and month of the time the nearest competitor was opened
# - Promo2: Promo2 is a continuing a promotion for some stores: 0 = store is not participating, 1 = store is participating
# - Promo2Since[Year/Week]: describes the year and calendar week when the store started participating in Promo2
# - PromoInterval: describes the consecutive intervals Promo2 is started, naming the months the promotion is started. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
# 

# ### Checking for missing values

# In[18]:


sns.heatmap(df_store.isnull())


# In[19]:


df_store.isna().sum()


# In[20]:


#Dropping the columns with many NaN values
df_store.drop(['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], axis=1, inplace=True)


# ### Merging the datasets

# In[21]:


df_merge = pd.merge(df_train, df_store, how = 'left', on = 'Store')


# In[22]:


df_merge.head(10)


# In[23]:


df_merge.shape


# ### Checking for missing values

# In[24]:


sns.heatmap(df_merge.isnull())


# In[25]:


df_merge.isna().sum()


# In[26]:


#Checking for duplicates after merging
df_merge.duplicated(subset=['Store', 'Date']).value_counts()


# ### Splitting the 'Date' column into 'Year', 'Month' & 'Day'

# In[27]:


df_merge['Date'] = pd.to_datetime(df_merge['Date'])
df_merge['Year'] = df_merge.Date.dt.year
df_merge['Month'] = df_merge.Date.dt.month
#df_merge['Month'] = df_merge['Month'].apply(lambda x: calendar.month_abbr[x])
df_merge['Day'] = df_merge.Date.dt.day
#df_merge['WeekOfYear'] = df_merge.Date.dt.isocalendar().week


# In[28]:


df_merge.head(10)


# In[29]:


df_merge['Month'] = df_merge['Month'].astype(str).str.zfill(2)
df_merge['Months'] = df_merge['Year'].astype(str) + " - " + df_merge['Month'].astype(str)
df_merge.head(10)


# ### Correlation Heatmap

# In[30]:


def heatmap_all(combined):
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(combined.corr(), annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=12);
heatmap_all(df_merge)


# ### Data Visualization

# In[31]:


#Year-Sales
Sales_Year = df_merge.groupby(['Year'])[['Sales']].sum()
Sales_Year


# In[32]:


Sales_Year = Sales_Year.reset_index(level=0)


# In[33]:


sns.set(rc = {'figure.figsize':(6,4)}, font_scale = 1.2)
sns.barplot(x='Year', y='Sales', data=Sales_Year, palette="mako")
plt.xlabel("Years")
plt.ylabel("Number of Sales")


# In[37]:


#Months-Sales
Sales_YearMonth = df_merge.groupby(['Months'])[['Sales', 'Customers']].sum()
Sales_YearMonth.sort_values('Months')


# In[38]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=Sales_YearMonth, x="Months", y="Sales").set(title="No. of Sales from 2013-15")
plt.xticks(rotation=60)
plt.xlabel("Months (2013-15)")
plt.ylabel("Number of Sales")
plt.show()


# In[39]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=Sales_YearMonth, x="Months", y="Customers").set(title="No. of Customers from 2013-15")
plt.xticks(rotation=60)
plt.xlabel("Months (2013-15)")
plt.ylabel("Number of Customers")
plt.show()


# In[40]:


#Months-Open
Open_YearMonth = df_merge.groupby(['Months'])[['Open']].sum()
Open_YearMonth.sort_values('Months')


# In[41]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=Open_YearMonth, x="Months", y="Open").set(title="No. of stores open from 2013-15")
plt.xticks(rotation=60)
plt.xlabel("Months (2013-15)")
plt.ylabel("Number of stores open")
plt.show()


# In[185]:


#Store_Types-Monthly_Sales
Type_YearMonth = df_merge.groupby(['Months', 'StoreType'])[['Sales']].sum()
Type_YearMonth.sort_values('Months')


# In[186]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=Type_YearMonth, x="Months", y="Sales", hue="StoreType")
plt.xticks(rotation=60)
plt.xlabel("Number of Sales")
plt.ylabel("Months (2013-15)")
plt.show()


# In[42]:


#Store_Type Count
sns.set(rc = {'figure.figsize':(7,4)}, font_scale = 1.2)
sns.countplot(x="StoreType", data=df_merge)
plt.xlabel("Store Types")
plt.ylabel("Store Count")


# In[44]:


#Months-Promo
sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
df_promo = df_merge.sort_values("Months")
sns.lineplot(data=df_promo, x="Months", y="Sales", hue="Promo")
plt.xticks(rotation=60)
plt.xlabel("Months (2013-15)")
plt.ylabel("Number of Sales")
plt.show()


# In[45]:


#Promo-SchoolHoliday
sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=df_promo, x="Months", y="Sales", hue="Promo", style='SchoolHoliday')
plt.xticks(rotation=60)
plt.xlabel("Months (2013-15)")
plt.ylabel("Number of Sales")
plt.show()


# In[46]:


#Sales-StoreType
sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=df_promo, x="Months", y="Sales", hue="StoreType", style='Promo')
plt.xticks(rotation=60)
plt.xlabel("Months (2013-15)")
plt.ylabel("Number of Sales")
plt.show()


# ### Encoding the Categorical Variables

# In[54]:


#Creating a dummy dataset and dropping 'Date' column
df_model = df_merge.copy()
df_model.drop(['Date'], axis=1, inplace=True)


# In[55]:


df_model.head(10)


# ### One-Hot Encoding

# In[107]:


cat_cols = df_model[['StoreType', 'Assortment']] 


# In[108]:


from sklearn.preprocessing import OneHotEncoder
#One-hot-encoding the categorical columns.
encoder = OneHotEncoder(handle_unknown='ignore')
#Converting it to dataframe
df_encoder = pd.DataFrame(encoder.fit_transform(cat_cols).toarray())
df_final = df_model.join(df_encoder)
df_final.head()


# In[109]:


#Dropping the columns that have already been One-Hot-Encoded
df_final.drop(['StateHoliday', 'StoreType', 'Assortment', 'Months', 'Customers'], axis=1, inplace=True)


# In[110]:


col = df_final.pop('Sales')
df_final.insert(0, 'Sales', col)
df_final.head()


# In[111]:


X = df_final.iloc[:, 1:].values
y = df_final.iloc[:, 0].values


# In[112]:


X


# In[113]:


y


# ### Splitting the dataset

# In[114]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)


# In[115]:


X_train


# In[116]:


X_test


# In[117]:


y_train


# In[118]:


y_test


# ### Random Forest Regression Model

# In[119]:


#Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# ### Predicting the Test set results

# In[120]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
pred = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
pred[:10, :]


# In[121]:


pred_plot = pd.DataFrame(pred, columns=['Predicted', 'Actual'])
pred_plot.head(15)


# ### Evaluating the model performance

# In[122]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[123]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)


# In[124]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# ### LightGBM Regression Model

# In[125]:


#Training the LightGBM model on the whole dataset
import lightgbm as lgb
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
expected_y  = y_test
predicted_y = model.predict(X_test)


# ### Evaluating the model performance

# In[126]:


#R-Squared
from sklearn import metrics
r2_score(expected_y, predicted_y)


# In[127]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mean_absolute_error(expected_y, predicted_y)


# In[128]:


#Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(expected_y, predicted_y)


# ### Assessing the 'test.csv' dataset

# In[129]:


df_test.head(10)


# In[130]:


df_test.shape


# In[131]:


df_test.info()


# There are 41088 rows with 8 features in the "test.csv" dataset, where 2 features are categorical (Date & StateHoliday) while the rest are numerical and contains the following fields:

# - Id: an Id that represents a (Store, Date) duple within the test set
# - Store: a unique Id for each store
# - Open: an indicator for whether the store was open: 0 = closed, 1 = open.
# - Promo: indicates whether a store is running a promo on that day.
# - StateHoliday: indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. All schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
# - SchoolHoliday: indicates if the (Store, Date) was affected by the closure of public schools.

# ### Merging the 'test.csv' dataset with 'stores.csv' dataset

# In[132]:


df_result = pd.merge(df_test, df_store, how = 'left', on = 'Store')


# In[133]:


df_result.shape


# ### Checking for missing values

# In[134]:


sns.set(rc = {'figure.figsize':(6,4)}, font_scale = 1)
sns.heatmap(df_result.isnull())


# In[135]:


df_result.isna().sum()


# In[136]:


# Replacing missing values in 'Open' column with '1.0'
df_result['Open'] = df_result['Open'].fillna(1.0)


# ### Splitting the 'Date' column into 'Year', 'Month' & 'Day'

# In[137]:


df_result['Date'] = pd.to_datetime(df_result['Date'])
df_result['Year'] = df_result.Date.dt.year
df_result['Month'] = df_result.Date.dt.month
df_result['Day'] = df_result.Date.dt.day


# In[138]:


df_result.head(10)


# ### Encoding the Categorical Columns for 'test.csv'

# In[139]:


# Dropping the 'Date' column
df_result.drop('Date', axis=1, inplace=True)


# ### One-Hot Encoding

# In[140]:


res_cols = df_result[['StoreType', 'Assortment']] 


# In[141]:


from sklearn.preprocessing import OneHotEncoder
#One-hot-encoding the categorical columns.
encoder = OneHotEncoder(handle_unknown='ignore')
#Converting it to dataframe
df_encoder = pd.DataFrame(encoder.fit_transform(res_cols).toarray())
df_res_final = df_result.join(df_encoder)
df_res_final.head()


# In[142]:


#Dropping the columns that have already been One-Hot-Encoded
df_res_final.drop(['StateHoliday', 'StoreType', 'Assortment', 'Id'], axis=1, inplace=True)


# ### Predicted Sales

# In[143]:


#Predicted Sales for the test set
predicted_test = regressor.predict(df_res_final)


# In[144]:


predicted_test


# In[145]:


pred_sales = pd.DataFrame(predicted_test, columns=['Predicted_Sales'])


# In[146]:


df_graph = pred_sales.copy()


# In[147]:


df_graph.head(10)


# ### Visualization of the Predicted Values

# In[156]:


df_graph["Year"] = df_result["Year"]
df_graph["Month"] = df_result["Month"]
df_graph["Day"] = df_result["Day"]


# In[157]:


df_graph['Day'] = df_graph['Day'].astype(str).str.zfill(2)
df_graph['Month-Day'] = df_graph['Month'].astype(str) + "-" + df_graph['Day'].astype(str)


# In[158]:


df_graph.drop(["Month", "Day"], axis=1, inplace=True)


# In[159]:


df_graph.head(10)


# In[160]:


Predict_Month = df_graph.groupby(['Month-Day'])[['Predicted_Sales']].sum()
Predict_Month


# In[161]:


Predict_Month['Predicted_Sales'] = Predict_Month['Predicted_Sales'].astype('int')
Predict_Month


# In[162]:


start_date = '2013-08-01'
end_date = '2013-09-17'


# In[163]:


mask_13 = (df_merge['Date'] >= start_date) & (df_merge['Date'] <= end_date)


# In[164]:


df_13 = df_merge.loc[mask_13]
df_13.head(10)


# In[165]:


df_13['Date'] = pd.to_datetime(df_13['Date'])
df_13['Year'] = df_13.Date.dt.year
df_13['Month'] = df_13.Date.dt.month
df_13['Day'] = df_13.Date.dt.day


# In[166]:


df_13['Day'] = df_13['Day'].astype(str).str.zfill(2)
df_13['Month-Day'] = df_13['Month'].astype(str) + "-" + df_13['Day'].astype(str)


# In[167]:


df_13.head(10)


# In[168]:


start_date = '2014-08-01'
end_date = '2014-09-17'


# In[169]:


mask_14 = (df_merge['Date'] >= start_date) & (df_merge['Date'] <= end_date)


# In[170]:


df_14 = df_merge.loc[mask_14]
df_14.head(10)


# In[171]:


df_14['Date'] = pd.to_datetime(df_14['Date'])
df_14['Year'] = df_14.Date.dt.year
df_14['Month'] = df_14.Date.dt.month
df_14['Day'] = df_14.Date.dt.day


# In[172]:


df_14['Day'] = df_14['Day'].astype(str).str.zfill(2)
df_14['Month-Day'] = df_14['Month'].astype(str) + "-" + df_14['Day'].astype(str)


# In[ ]:


df_14.head(10)


# In[178]:


#Creating a table with the sales from every year
compare = df_13.groupby(['Month-Day'])[['Sales']].sum()
compare['2013'] = compare['Sales']
compare.head(10)


# In[179]:


compare.drop('Sales', axis=1, inplace=True)


# In[180]:


compare['2013'] = compare['2013'].astype('int')
compare


# In[181]:


compare['2014'] = df_14.groupby(['Month-Day'])[['Sales']].sum()
compare['2014'] = compare['2014'].astype('int')
compare


# In[182]:


compare['2015'] = Predict_Month['Predicted_Sales']
compare


# In[184]:


#Plot for the comparison of sales
sns.set(rc = {'figure.figsize':(30,15)}, font_scale = 2.5)
sns.lineplot(data=compare, x="Month-Day", y="2013", color='r', label='Sales in 2013').set(title="Comparison of Predicted Sales & Previous Years' Sales")
sns.lineplot(data=compare, x="Month-Day", y="2014", color='b', label='Sales in 2014')
sns.lineplot(data=compare, x="Month-Day", y="2015", color='g', label='Predicted Sales in 2015')
plt.xticks(rotation=60)
plt.xlabel("Months")
plt.ylabel("Number of Sales")
plt.show()


# The above graph shows the comparison of the predicted sales values in the year 2015 and sales numbers in the years 2013 & 2014. The predicted sales values were very similar to the previous years’ sales values that means if there would have been data about the actual sales values from the ‘test.csv’ dataset, it would have been very close.

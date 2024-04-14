#!/usr/bin/env python
# coding: utf-8

# ## Importing dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')


# ## Reading Data

# In[2]:


df = pd.read_csv(f'Flipkart.csv')
df


# ## Data Overview (Pre-Analysis)

# In[3]:


null_counts = df.isnull().sum()
non_null_counts = df.notnull().sum()

# Create a DataFrame for plotting
plot_data = pd.DataFrame({'Null': null_counts, 'NonNull': non_null_counts})
plot_data.plot(y='Null',kind='bar',)
plt.title('Null Values Count')
plt.xlabel('Columns')
plt.ylabel('Count')
plt.show()


# ### 1. Removing redundant columns (Data Analysis Technique - 1)
# Redundant index columns merely reflect the row numbers or identifiers and do not contribute meaningful information to the analysis. By eliminating these columns, the dataset becomes more streamlined, enhancing clarity and simplifying interpretation. 

# In[3]:


df = df.iloc[:, 1:]


# In[4]:


df.info()


# ### 2. Checking for null values (Data Analysis Technique - 2)
# Since our data has multiple columns, before proceeding with any other data processing steps, it is crucial to check for null values in the data 

# In[5]:


df.isnull().sum()


# ### 3. Filling null values (Data Analysis Technique - 3)
# Filling NA (missing) values in a dataset is essential to ensure the completeness and accuracy of the data for analysis. By filling NA values, we prevent the loss of valuable information that could affect the integrity of the analysis results. Moreover, it helps in maintaining consistency across the dataset, ensuring that subsequent operations such as statistical analysis or machine learning modeling can be performed effectively without encountering errors or biases due to missing data. 

# In[6]:


for col in ['Bluetooth Support', 'Wi-Fi', 'Audio Jack', 'Domestic Warranty', 'Network Type', 'Primary Camera Available', 'Secondary Camera Available', 'Full HD Recording', 'GPS Support', 'RAM'] :
    print(f'Unique values in {col} are {df[col].unique()}')


# Using domain knowledge about recent smartphones -
# - We know most smartphones nowadays have a bluetooth support. Thus, we are replacing the nan values in the column with 'Yes'.
# - We know most smartphones nowadays have WiFi support. Thus, we are replacing the nan values in the column with 'Yes'.
# - We know most smartphones nowadays do not have a Audio Jack WiFi support. Thus, we are replacing the nan values in the column with 'No'.
# - We know that smartphones nowadays have 2G, 3G, 4G connectivitity.
# - We know that smartphones nowadays have primary camera.
# - We know that smartphones nowadays have secondary camera.

# In[7]:


df_cleaned = df.copy()


# In[8]:


df_cleaned = df_cleaned.dropna(subset='Name')


# In[9]:


df_cleaned['Bluetooth Support'] = df_cleaned['Bluetooth Support'].fillna('Yes')


# In[10]:


df_cleaned['Wi-Fi'] = df_cleaned['Wi-Fi'].fillna('Yes')


# In[11]:


df_cleaned['Audio Jack'] = df_cleaned['Audio Jack'].fillna('Yes')


# In[12]:


df_cleaned['Primary Camera'] = df_cleaned['Audio Jack'].fillna('Yes')


# In[13]:


df_cleaned['Network Type'] = df_cleaned['Network Type'].fillna('2G, 3G, 4G')


# In[14]:


df_cleaned['Primary Camera Available'] = df_cleaned['Primary Camera Available'].fillna('Yes')


# In[15]:


df_cleaned['Secondary Camera Available'] = df_cleaned['Secondary Camera Available'].fillna('Yes')


# For the columns [`Full HD Recording`, `GPS Support`, `Network Type`, `Battery Capacity`, `Processor Core`, `Primary Clock Speed`, `Primary Camera`] we fill the null values by the most frequently occuring value (or the mode) in the column since it is a better representative of trends in majority of smartphones.

# In[16]:


mode_full_hd_recording = df_cleaned['Full HD Recording'].value_counts().index[0]
df_cleaned['Full HD Recording'] = df_cleaned['Full HD Recording'].fillna(mode_full_hd_recording)


# In[17]:


mode_gps_support = df_cleaned['GPS Support'].value_counts().index[0]
df_cleaned['GPS Support'] = df_cleaned['GPS Support'].fillna(mode_gps_support)


# In[18]:


df_cleaned = pd.concat((df_cleaned, df_cleaned['Network Type'].str.get_dummies(sep=', ')), axis=1)
df_cleaned = df_cleaned.drop(columns=['Network Type'])


# In[19]:


mode_battery_capacity = df_cleaned['Battery Capacity'].value_counts().index[0]
df_cleaned['Battery Capacity'] = df_cleaned['Battery Capacity'].fillna(mode_battery_capacity)


# In[20]:


mode_processor_core = df_cleaned['Processor Core'].value_counts().index[0]
df_cleaned['Processor Core'] = df_cleaned['Processor Core'].fillna(mode_processor_core)


# In[21]:


mode_clock_speed = df_cleaned['Primary Clock Speed'].value_counts().index[0]
df_cleaned['Primary Clock Speed'] = df_cleaned['Primary Clock Speed'].fillna(mode_clock_speed)


# In[22]:


mode_primary_camera = df_cleaned['Primary Camera'].value_counts().index[0]
df_cleaned['Primary Camera'] = df_cleaned['Primary Camera'].fillna(mode_primary_camera)


# ### 4. Feature Transformation (Data Analysis Technique - 4)
# We extract the numerical information from columns containing strings so that better analysis techniques can be applied to the numerical data like mean, standard deviation

# In `Battery Capacity` column, since each value is in mAh it is redundant to be included in each record, thus we extract only the numerical value of the battery size.

# In[23]:


df_cleaned['Battery Capacity'] = df_cleaned['Battery Capacity'].str.extract('(\d+)').astype(float)


# In `Price` colummn, since each value is in ₹  it is redundant to be included in each record, thus we remove the currency symbol and commas from the column and extract only the numerical price. 

# In[24]:


df_cleaned['Price'] = df_cleaned['Price'].str.replace('[^\d]', '', regex=True).astype(int)


# In `Weight` column, since each value is in g(grams) it is redundant to be included in each record, thus we remove the weight unit symbol and extract only the numerical weight. Further, we fill the null values with the mean of the column.

# In[25]:


df_cleaned['Weight'] = df_cleaned['Weight'].str.extract('(\d+)').astype(float)
df_cleaned['Weight'] = df_cleaned['Weight'].fillna(df_cleaned['Weight'].mean())


# In `Domestic Warranty` column, we map the unique string values with numerbers equivalent to the months of warrants. We fill the null values with a warranty of 0.

# In[26]:


df_cleaned['Domestic Warranty'].unique()


# In[27]:


mapping = {'1 Year': 12, '0': 0, '12 Months': 12, '2 Year': 24, np.nan: 0}

df_cleaned['Domestic Warranty']= df_cleaned['Domestic Warranty'].map(mapping)


# In columns [`Internal Storage`, `RAM`], we convert the data into numerical figures representing the values in GB since it is a standard for smarphones RAM and storage be measured in GB. Further, we fill the null values with the mean of the column.

# In[28]:


def convert_to_gb(value):
    if isinstance(value, str) :
        if 'GB' in value:
            return float(value.replace(' GB', ''))
        elif 'MB' in value:
            return float(value.replace(' MB', '')) / 1024  # Convert MB to GB
        else:
            return 0  # Handle other cases like '0 GB', '0 MB', etc.
    else :
        return np.nan


# In[29]:


df_cleaned['Internal Storage'] = df_cleaned['Internal Storage'].apply(convert_to_gb).apply(float)

df_cleaned.rename(columns={'Internal Storage': 'Internal Storage (GB)'}, inplace=True)

df_cleaned['Internal Storage (GB)'] = df_cleaned['Internal Storage (GB)'].fillna(df_cleaned['Internal Storage (GB)'].mean())


# In[30]:


df_cleaned['RAM'] = df_cleaned['RAM'].apply(convert_to_gb).astype(float)

df_cleaned.rename(columns={'RAM': 'RAM (GB)'}, inplace=True)


# In `Secondary Camera`, since each value is in 'xMP Front Camera' it is redundant to be included in each record, thus we remove the 'MP Front Camera' part and extract only the numerical value of the MP. Further, we fill the null values with the mean of the column.

# In[31]:


df['Secondary Camera'].value_counts()


# In[32]:


df_cleaned['Secondary Camera'] = df_cleaned['Secondary Camera'].str.extract('(\d+)').astype(float)
df_cleaned['Secondary Camera'] = df_cleaned['Secondary Camera'].fillna(df_cleaned['Secondary Camera'].mean()).astype(int)
df_cleaned.rename(columns={'Secondary Camera' : 'Secondary Camera (MP, Front Camera)'}, inplace=True)


# In[33]:


sim_mapping = {'Dual Sim' : 2, 'Single Sim' : 1, 'Dual Sim(Nano + eSIM)' : 0}
df_cleaned['SIM Type'] = df_cleaned['SIM Type'].map(sim_mapping)


# In[34]:


processor_mapping = {'Octa Core' : 8, 'Hexa Core' : 6, 'Quad Core' : 4, 'Single Core' : 1}
df_cleaned['Processor Core'] = df_cleaned['Processor Core'].map(processor_mapping)


# In[35]:


df_cleaned['Display Size (cm)'] = df_cleaned['Display Size'].str.extract('r(\d+\.\d+) cm')[0].astype(float)
df_cleaned = df_cleaned.drop(columns='Display Size')


# ### 5. Feature Generation (Data Analysis Technique - 5)
# This feature generation technique is employed to transform unstructured textual data into a structured numerical format, primarily for enhanced analysis and interpretation. By extracting numeric values from the 'Resolution' column and splitting them into separate 'Height' and 'Width' columns, the dataset becomes more organized and suitable for various analytical tasks. Handling missing values by filling them with the mode ensures the robustness of subsequent operations and maintains the dataset's representativeness. Converting data types to integers facilitates numerical computations and statistical analyses, enabling researchers to derive meaningful insights from the data. Additionally, dropping the original 'Resolution' column streamlines the dataset, eliminating redundancy and improving computational efficiency. 

# In[36]:


df_cleaned['Resolution'] = df_cleaned['Resolution'].str.extract('(\d+ x \d+)')
mode_resolution = df_cleaned['Resolution'].value_counts().index[0]
df_cleaned['Resolution'] = df_cleaned['Resolution'].fillna(mode_resolution)
df_cleaned[['Height', 'Width']] = df_cleaned['Resolution'].str.split(' x ', expand=True)
df_cleaned[['Height', 'Width']] = df_cleaned[['Height', 'Width']].astype(int)
df_cleaned = df_cleaned.drop(columns=['Resolution'])


# In[37]:


os_map = {'Android' : 0, 'iPhone' : 1}
iphone_regex = r'iPhone'
df_cleaned['OS'] = df_cleaned['Name'].str.contains(iphone_regex, case=False).astype(np.int32)
df_cleaned = df_cleaned.drop(columns='Name')


# ### 6. Binary Encoding (Data Analysis Technique - 6)
# We convert the remaining binary categorical features into numerical binaros_mapencoded features where 1 - 'Yes' and 0 - 'No'.
# 
# By converting categorical variables like 'Yes' and 'No' into numerical equivalents (1 and 0, respectively), we transform the data into a format that can be easily understood and processed by algorithms. This transformation enables us to perform various statistical analysis, build predictive models, and extract meaningful insights from the data.
# 
# Furthermore, replacing categorical values with numerical equivalents helps in standardizing the data and making it consistent across different variables, which is crucial for ensuring the effectiveness of machine learning algorithms.

# In[38]:


df_cleaned = df_cleaned.replace({'Yes': 1, 'No': 0})


# In[39]:


df_cleaned


# In[160]:


df_cleaned.isnull().sum()


# In[161]:


df_cleaned.info()


# ### 7. Correlation Analysis (Data Analysis Technique - 7)
# We employed correlation analysis to investigate the linear relationship between numerical variables in the dataset. By calculating the correlation coefficients between pairs of variables using methods like Pearson correlation, we gain insights into how changes in one variable might relate to changes in another. This analysis helps in identifying potential associations or dependencies between variables, which is valuable for tasks such as feature selection, model building, and understanding the underlying structure of the data. 

# In[38]:


corr = df_cleaned.select_dtypes(include=[int, float]).corr()
corr


# In[162]:


plt.figure(figsize=(18, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')


# ### 8. Data Visualisations (Data Analysis Technique - 8)
# We use different visualisations like histogram plot, box plot, scatter plot, pie chart to find the distribution of data values

# In[208]:


sns.pairplot(df_cleaned.select_dtypes(['float']))


# In[40]:


def plot_distribution(df : pd.Series) :
    min_value = df.min()
    max_value = df.max()
    mean_value = df.mean()
    median_value = df.median()
    mode_values = df.mode()
    
    print(f'The minimum value is {min_value:.2f}\nThe maximum value is {max_value:.2f}\nThe mean value is {mean_value:.2f}\nThe median value is {median_value:.2f}\nThe mode values are {mode_values.values}')
    
    fig, ax = plt.subplots(2, 1, figsize = (8,6))

    #ax[0].hist(df, alpha = 0.5)
    #sns.kdeplot(df, ax=ax[0], linewidth = 2)
    #sns.histplot(df, ax=ax[0], kde = True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.distplot(df, ax=ax[0])
    ax[0].set_ylabel('Density')

    ax[0].axvline(x=min_value, color = 'gray', linestyle='dashed', linewidth = 1, label = 'Min', alpha=0.5)
    ax[0].axvline(x=mean_value, color = 'cyan', linestyle='dashed', linewidth = 1, label = 'Mean', alpha=0.5)
    ax[0].axvline(x=median_value, color = 'red', linestyle='dashed', linewidth = 1, label = 'Median', alpha=0.5)
    for mode in mode_values :
        ax[0].axvline(x=mode, color = 'yellow', linestyle='dashed', linewidth = 1, label = 'Mode', alpha=0.5)
    ax[0].axvline(x=max_value, color = 'gray', linestyle='dashed', linewidth = 1, label = 'Max', alpha=0.5)
    ax[0].legend()

    ax[1].boxplot(df, vert=False)
    ax[1].set_xlabel('Value')

    fig.suptitle(df.name)
    plt.grid()
    plt.tight_layout()
    plt.show()


# In[41]:


plot_distribution(df_cleaned['Price'])


# In[42]:


plot_distribution(df_cleaned['Height'])


# In[43]:


plot_distribution(df_cleaned['Width'])


# In[44]:


plot_distribution(df_cleaned['Weight'])


# In[45]:


plot_distribution(df_cleaned['Battery Capacity'])


# In[46]:


plot_distribution(df_cleaned['RAM (GB)'])


# In[47]:


plot_distribution(df_cleaned['Internal Storage (GB)'])


# In[48]:


sns.scatterplot(df_cleaned, x='Width', y='Height', hue='Price')


# In[60]:


sim_labels = list(sim_mapping.keys())
plt.figure(figsize=(5, 5))
df_cleaned['SIM Type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, labels=sim_labels)
plt.title('Distribution of SIM Types')
plt.tight_layout()
plt.ylabel('');


# In[99]:


processor_labels = list(processor_mapping.keys())
plt.figure(figsize=(5, 5))
df_cleaned['Processor Core'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, labels=processor_labels)
plt.title('Distribution of Processor Cores')
plt.tight_layout()
plt.ylabel('');


# In[51]:


df_cleaned['5G'].value_counts().rename({1: '5G', 0: 'Non-5G'}).plot.pie(autopct='%1.1f%%', startangle=140, labels=['5G', 'Non-5G'])
plt.title('Smarthphones with 5G connectivity')
plt.tight_layout()
plt.ylabel('');


# In[121]:


df_cleaned['OS'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, labels=list(os_map.keys()))
plt.title('Distribution of OS')
plt.tight_layout()
plt.ylabel('');


# ## Predicting Price

# ### Seperating independent and dependent features

# In[182]:


dependent = 'Price'
independent = df_cleaned.select_dtypes(['int', 'float', 'bool']).iloc[:, 1:].columns
# independent = corr[dependent][corr[dependent].abs() > 0.1][1:].index
print(f'Dependent feature is {dependent}.\nIndependent features are {independent}')


# ### Train Test Split

# In[183]:


X_train, X_test, y_train, y_test = train_test_split(df_cleaned[independent], df_cleaned[dependent], train_size=0.8, random_state=0)
print(f'X_train.shape = {X_train.shape}\nX_test.shape = {X_test.shape}\ny_train.shape = {y_train.shape}\ny_test.shape = {y_test.shape}')


# ## 1. Linear Regression
# ### Training

# In[184]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[185]:


y_pred = model.predict(X_test)


# ### Evaluating Predictions

# In[186]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


# ### Visualising Predictions 

# In[187]:


plt.scatter(y_test, y_pred, s = 13)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')


#Fline for perfect correlation
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='yellow', linewidth=3, label = 'Using numpy')

#Second method- Again applying Linear regression
lr2 = LinearRegression()
y_test_2 = np.array(y_test).reshape(-1, 1)
y_pred_2 = y_pred.reshape(-1, 1)
lr2.fit(y_test_2, y_pred_2)
plt.plot(y_test_2, lr2.predict(y_test_2), color='red', alpha = 0.5, label = 'Using LinReg')
plt.legend(title = 'Perfect Correlation Line')

#2 lines will coincide
#This fitted line helps visualize how well the predicted values align with the actual values.


# ## 2. Random Forest Regression
# ### Training

# In[188]:


rfg = RandomForestRegressor()
rfg.fit(X_train, y_train)


# In[189]:


y_pred = rfg.predict(X_test)


# ### Evaluating Predictions

# In[190]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


# ### Visualising Predictions

# In[191]:


plt.scatter(y_test, y_pred, s = 13)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')


#Fline for perfect correlation
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='yellow', linewidth=3, label = 'Using numpy')

#Second method- Again applying Linear regression
lr2 = LinearRegression()
y_test_2 = np.array(y_test).reshape(-1, 1)
y_pred_2 = y_pred.reshape(-1, 1)
lr2.fit(y_test_2, y_pred_2)
plt.plot(y_test_2, lr2.predict(y_test_2), color='red', alpha = 0.5, label = 'Using LinReg')
plt.legend(title = 'Perfect Correlation Line')

#2 lines will coincide
#This fitted line helps visualize how well the predicted values align with the actual values.


# ## 3. Gradient Boosting Regression
# ### Training

# In[196]:


gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)


# In[197]:


y_pred = gbr.predict(X_test)


# ### Evaluating Predictions

# In[198]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


# ### Visualising Predictions

# In[199]:


plt.scatter(y_test, y_pred, s = 13)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')


#Fline for perfect correlation
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='yellow', linewidth=3, label = 'Using numpy')

#Second method- Again applying Linear regression
lr2 = LinearRegression()
y_test_2 = np.array(y_test).reshape(-1, 1)
y_pred_2 = y_pred.reshape(-1, 1)
lr2.fit(y_test_2, y_pred_2)
plt.plot(y_test_2, lr2.predict(y_test_2), color='red', alpha = 0.5, label = 'Using LinReg')
plt.legend(title = 'Perfect Correlation Line')

#2 lines will coincide
#This fitted line helps visualize how well the predicted values align with the actual values.


# In[ ]:





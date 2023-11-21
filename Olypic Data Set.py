#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os


# In[ ]:


# In this code snippet, several Python libraries are imported. Pandas is used for data manipulation and analysis, Seaborn for statistical data visualization, NumPy for numerical operations, and Matplotlib for creating plots and graphs. Warnings are managed to handle issues during execution, and the os module can be used for file operations and directory handling.


# In[2]:


get_ipython().system('pip install pandas-profiling')


# In[ ]:


#"Pandas-Profiling" is used for automatic exploratory data analysis (EDA) in Python.


# In[2]:


df=pd.read_csv('Book2.csv')
df


# In[ ]:


# The given code reads data from a CSV file named "Book2.csv" and creates a DataFrame named 'df', which allows easy manipulation, analysis, and visualization of tabular data in Python using the pandas library.


# In[6]:


df.head() #show top 5 Records


# In[3]:


df['Sport'].value_counts()


# In[7]:


df.tail()  #show Down Top 5 Records


# # Data Preprocessing Part 1

# In[8]:


df.nunique()

# The DataFrame contains data from various Olympic Games, with 7,101 unique IDs for athletes. It includes 4,950 unique names, with a nearly equal distribution of genders (2 unique values). The athletes' ages range from young to old (43 unique values). Height varies across 77 unique values, and weight spans 118 unique values. The dataset covers 173 teams from 114 National Olympic Committees. Across 50 games held over 35 years, there were both summer and winter seasons (2 unique values). The games took place in 42 different cities, involving 54 distinct sports and 495 event types. Athletes received three types of medals for their performances.
# In[14]:


df.info()

# The DataFrame contains 7,101 rows and 15 columns, with no missing data (non-null counts are all 7101). The dataset includes information on athletes' ID, name, sex, age, height, weight, team, National Olympic Committee (NOC), games, year, season, city, sport, event, and medal type. Most of the columns are of object data type, with four integer columns and one float column. The data appears to be well-structured and ready for analysis.
# In[23]:


df.shape


# In[16]:


df.size                   #Total Size Of Data Set


# In[7]:


df['Team'].value_counts()


# In[18]:


df.(['India']).head()


# In[22]:


df.columns              #how many columns in this Data Sets


# In[13]:


df.describe()


# In[ ]:


# The `df.describe()` output provides a comprehensive statistical analysis of the DataFrame 'df.' It includes count, mean, standard deviation, minimum, 25th percentile (Q1), median (50th percentile or Q2), and 75th percentile (Q3) values for columns "ID," "Age," "Height," "Weight," and "Year." The "ID" column ranges from 1 to 7101, representing unique identifiers. The "Age" column has an average of 25.46 years, with a minimum of 13 and a maximum of 58. The "Height" column ranges from 136 to 223, with an average of approximately 178 cm. The "Weight" column ranges from 30 to 170 kg, with an average of approximately 73.93 kg. The "Year" column spans from 1896 to 2016, with an average year of approximately 1988.12. This summary aids in understanding the data's central tendencies, spread, and overall distribution across the mentioned numerical columns.


# In[15]:


df[df.duplicated()]   #This code filters and returns rows in the DataFrame 'df' that are duplicates based on all columns.


# In[34]:


df.profile_report(style={'full_widh':True})


# In[17]:


df.isnull().sum()   #null Values


# In[19]:


df.isnull().mean()*100     #The code calculates the percentage of missing values in each column of the DataFrame 'df'.


# In[24]:


df.std()


# In[ ]:


#The `df.std()` function calculates the standard deviation, a measure of data spread, for each numerical column in the DataFrame 'df.' The results show the amount of variation in each attribute. The "ID" column exhibits considerable variability with a standard deviation of approximately 2050, while the "Age," "Height," "Weight," and "Year" columns also demonstrate notable diversities, providing insights into athletes' characteristics and event years in the dataset.


# In[26]:


df.mean()


# In[ ]:


#The `df.mean()` function calculates the mean value for each numerical column in the DataFrame 'df.' The results provide insights into the central tendencies of the data. The "ID" column has an average value of 3551, representing the identifiers. The "Age," "Height," and "Weight" columns show average ages, heights, and weights of approximately 25.46 years, 178.00 cm, and 73.93 kg, respectively. The "Year" column indicates an average Olympic event year of around 1988.12.


# In[29]:


df['Team'][:5]   


# #The given code displays the first five entries of the 'Team' column in the DataFrame 'df', showing the national teams' names for athletes. It provides an initial view of the 'Team' data, indicating that the team name 'Finland' appears repeatedly in the first five rows.

# In[31]:


df.Age.describe()


# In[ ]:


# The statistical report for the 'Age' column in the DataFrame 'df' contains descriptive measures. The column has 7,101 non-null values, with a mean age of approximately 25.46 years. The age distribution shows a standard deviation of approximately 5.12 years. The youngest athlete is 13 years old, while the oldest is 58 years old. The 25th percentile indicates that 25% of athletes are 22 years or younger, the median (50th percentile) age is 25 years, and the 75th percentile shows that 25% of athletes are 28 years or older.


# In[34]:


df.Age.count()    #Total Count


# In[36]:


df.groupby('Age')[['Height']].mean()


# In[ ]:


#The statistical report analyzes the 'Height' column based on different age groups. The mean height increases steadily from 13 to 21, reaching approximately 178 cm. From ages 22 to 28, the height remains relatively stable around 178-179 cm. Afterward, the heights fluctuate with some variations. Notably, there are extreme values, like 188 cm for age 55 and 156 cm for age 58, which might be outliers.


# In[41]:


df[12:15]             #deffine 12 to 15 Records Only


# In[57]:


df.skew()


# In[ ]:


#The `df.skew()` function calculates the skewness of each column in the DataFrame 'df'. Skewness measures the asymmetry of the data distribution. Positive values indicate a right-skewed distribution, while negative values indicate a left-skewed distribution. In this dataset, 'Age' has a positive skew, 'Height' and 'Weight' have negative skew, and 'ID' and 'Year' are approximately normally distributed.


# In[58]:


df.kurt()


# In[ ]:


#The `df.kurt()` function calculates the kurtosis of each column in the DataFrame 'df'. Kurtosis measures the peakedness or flatness of a distribution compared to the normal distribution. The 'Age' column exhibits leptokurtic behavior with high kurtosis, while the 'ID,' 'Height,' 'Weight,' and 'Year' columns have platykurtic distributions with lower kurtosis values, indicating flatter tails compared to a normal distribution.


# In[63]:


df.corr()


# In[ ]:


#The correlation matrix shows the relationships between numeric columns in the DataFrame 'df'. 'Age' has a negative correlation with 'ID' and 'Year,' indicating that younger athletes tend to have higher ID numbers and participate in more recent Olympic events. There is a strong positive correlation between 'Height' and 'Weight,' suggesting that taller athletes tend to have higher weights. Additionally, 'Height' and 'Year,' as well as 'Weight' and 'Year,' show positive correlations, indicating that both height and weight of athletes have increased over the years.


# In[65]:





# In[10]:


plt.hist(df['Age'] ,bins=8, density=1)


# In[47]:


sns.distplot(df[('Age')])


# In[49]:


df.groupby(['City'])['Age'].count().plot(kind='bar')


# In[53]:


sns.boxplot(x='Team',y='Age', data=df)


# In[29]:


sns.heatmap(df.corr(method = 'spearman'), annot=True)


# In[19]:


plt.bar(x = df[''],height = df[''])
plt.title('Distribution of Companies Market Value by Continent')
plt.xlabel('Age')
plt.ylabel('City')
plt.show()


# In[17]:


sns.barplot(y= df.Age, x = df .Age)
plt.show()


# In[22]:


sns.scatterplot(df.Age)
plt.show()


# In[25]:


boxplt = sns.boxplot(df.Height)
plt.show()


# In[33]:


plt.pie(quantity,labels=Age)
plt.show()


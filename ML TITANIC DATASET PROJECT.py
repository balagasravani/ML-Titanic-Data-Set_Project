#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyforest


# In[1]:


data=pd.read_csv(r"C:\Users\srava\Downloads\Programs\Titanic-Dataset.csv")
data


# In[2]:


data.shape  #checking the how many rows and columns


# In[3]:


data.isna().sum()   #checking the how many null values present in dataset


# In[4]:


data.describe()


# In[5]:


data.info() #total information in dataset


# In[6]:


data.dtypes


# In[7]:


# import label encoder  #word sex converting as numeric
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder=preprocessing.LabelEncoder()

# Encode lebels in column 'Gender'.
data['Sex']= label_encoder.fit_transform(data['Sex'])

data['Sex'].value_counts() #converting male as 1 and female as 2


# In[8]:


data


# In[10]:


data=data.drop(['Ticket', 'Cabin', 'Name'],axis=1) #droping the unwanted columns
data


# In[11]:


data['Age'].median() # counting the how many null value in age


# In[12]:


data['Age']=data['Age'].fillna(value=28) # filling the null values using fillna
data


# In[13]:


data['Age'].isna().sum() 


# In[ ]:


get_ipython().system('pip install_profilling')


# In[14]:


data.isna().sum() 


# In[15]:


data['Embarked'].value_counts()


# In[16]:


g=data.groupby('Survived')
g['Embarked'].value_counts() #group by multple columns


# In[18]:


data['Embarked']=data['Embarked'].fillna(value='S') # filling the null values using fillna
data


# In[19]:


# import label encoder  #word sex converting as numeric
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder=preprocessing.LabelEncoder()

# Encode lebels in column 'Gender'.
data['Embarked']= label_encoder.fit_transform(data['Embarked'])

data['Embarked'].value_counts() #converting male as 1 and female as 2


# In[34]:


# countplot and catplot
sns.catplot(x='Embarked', hue='Survived', data=data, kind='count')
plt.title('Survival by Embarked')
plt.show()


# In[40]:


data.corr()


# In[43]:


data.plot(x='Survived', y=['SibSp','Parch'], kind='bar')
plt.show()


# In[51]:


sns.countplot(x='Sex', hue='Survived', data=data)
plt.show()


# In[52]:


correlation=data.corr()
correlation['Survived'].sort_values(ascending=False)


# In[53]:


sns.heatmap(data.corr())


# In[55]:


correlation['Fare'].sort_values(ascending=False)
correlation['Fare']


# In[56]:


data.head(8)


# In[57]:


data['Family']=data['SibSp']+data['Parch']+1
data=data.drop(['SibSp','Parch'],axis=1)
data=data.drop('Embarked', axis=1)
data


# In[58]:


data


# In[64]:


data['Fare']=pd.qcut(data['Fare'],4)
sns.barplot(x='Fare', y='Survived',data=data)


# In[66]:


sns.boxplot(x='Pclass', y='Age',data=data,hue='Survived')


# In[ ]:





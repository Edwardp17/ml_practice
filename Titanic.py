
# coding: utf-8

# # Solving the Titanic Problem - Machine Learning Classification
# 
# @Edwardp17
# 
# [Description of project, notebook, analysis, etc.]
# 
# Resources:
# - [Link Text](url)
# - [Link Text](url)
# - [Link Text](url)

# ## Import Libraries

# In[4]:

# general
import pandas as pd


# ## Read in Data

# In[3]:

df = pd.read_csv('train.csv')
df; #a `;` can be added to an output to suppress it from actually showing up in the notebook.


# ## Exploratory Analysis

# In[3]:

pd.isnull(df).any()


# In[4]:

df.head()


# In[5]:

df.dropna()


# In[11]:

df[df['Age']>30]


# In[14]:

df[df['Sex']=='female']


# In[15]:

df[(df['Age']>30) & (df['Sex']=='female')]


# In[17]:

femaleover30 = df[(df['Age']>30)&(df['Sex']=='female')]
femaleover30.describe()


# In[18]:

femaleover30survived = df[(df['Age']>30)&(df['Sex']=='female')&(df['Survived']==1)]


# In[19]:

femaleover30survived.describe()


# In[25]:

df.groupby(['Sex','Survived']).size()


# In[21]:

femaleover30survived.isnull().any()


# In[22]:

femaleover30survived.dropna()


# In[27]:

by_gender = df.groupby(['Sex','Survived']).size().unstack()
by_gender


# In[40]:

by_gender1 = by_gender.stack().to_frame().reset_index().rename(columns={0:"Count"})


# In[41]:

by_gender1


# In[ ]:




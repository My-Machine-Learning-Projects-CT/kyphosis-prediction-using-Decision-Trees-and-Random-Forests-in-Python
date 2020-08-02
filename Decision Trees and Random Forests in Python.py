#!/usr/bin/env python
# coding: utf-8


# # Decision Trees and Random Forests in Python

# This is the code for the lecture video which goes over tree methods in Python. Reference the video lecture for the full explanation of the code!
# 
#
# ## Import Libraries

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data

# In[8]:


df = pd.read_csv('kyphosis.csv')


# In[21]:


df.head()


# ## EDA
# 
# We'll just check out a simple pairplot for this small dataset.

# In[27]:


sns.pairplot(df,hue='Kyphosis',palette='Set1')


# ## Train Test Split
# 
# Let's split up the data into a training set and a test set!

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# ## Decision Trees
# 
# We'll start just by training a single decision tree.

# In[10]:


from sklearn.tree import DecisionTreeClassifier


# In[11]:


dtree = DecisionTreeClassifier()


# In[16]:


dtree.fit(X_train,y_train)


# ## Prediction and Evaluation 
# 
# Let's evaluate our decision tree.

# In[17]:


predictions = dtree.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix


# In[19]:


print(classification_report(y_test,predictions))


# In[20]:


print(confusion_matrix(y_test,predictions))


# ## Tree Visualization
# 
# Scikit learn actually has some built-in visualization capabilities for decision trees, you won't use this often and it requires you to install the pydot library, but here is an example of what it looks like and the code to execute this:

# In[33]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features


# In[39]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  


# ## Random Forests
# 
# Now let's compare the decision tree model to a random forest.

# In[41]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[45]:


rfc_pred = rfc.predict(X_test)


# In[46]:


print(confusion_matrix(y_test,rfc_pred))


# In[47]:


print(classification_report(y_test,rfc_pred))


# # Great Job!

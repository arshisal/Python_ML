#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[36]:


df_org = load_breast_cancer()


# In[37]:


type(df_org)


# In[38]:


df_org.keys()


# In[39]:


print(df_org['DESCR'])


# In[40]:


df = pd.DataFrame(df_org['data'], columns=df_org['feature_names'])


# In[41]:


df.head()


# In[20]:


scaler = StandardScaler()


# In[21]:


scaler.fit(df)


# In[22]:


scaled_df = scaler.transform(df)


# In[25]:


from sklearn.decomposition import PCA


# In[26]:


pca = PCA(n_components=3)


# In[27]:


pca.fit(scaled_df)


# In[28]:


x_pca = pca.transform(scaled_df)


# In[30]:


x_pca.shape


# In[42]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],cmap='plasma', c=df_org['target'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[43]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,2],cmap='plasma', c=df_org['target'])
plt.xlabel('First principal component')
plt.ylabel('Third Principal Component')


# In[53]:


plt.figure(figsize=(10,10))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x_pca[:,0],x_pca[:,1], x_pca[:,2],cmap='plasma', c=df_org['target'])
ax.set_xlabel('First principal component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')


# In[54]:


pca.components_


# In[55]:


df_comp = pd.DataFrame(pca.components_,columns=df_org['feature_names'])


# In[56]:


plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


# In[57]:


PC_values = np.arange(pca.n_components_) + 1


# In[59]:


pca.explained_variance_ratio_


# In[60]:


plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()


# In[ ]:





# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %%
# load data
df = pd.read_csv('house_votes_Dem.csv', encoding='latin-1')
# Trying to understand voting patterns 
# Unsupervised learning, there is no target variable
# salary is the variable in the lab
# %%
# take a look at the data
df.info()
# %%
# separate out the numeric features
c_num = df[['aye', 'nay', 'other']]
# %%
# documentation for kmeans in sklearn
#help(KMeans)
# %% build a kmeans model
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(c_num)


# %% look at the information in the model
print(kmeans.cluster_centers_)
print(kmeans.labels_)

# %%
# add the cluster labels to the original data frame
df['cluster'] = kmeans.labels_

# %%
  
# %% simple plot of the clusters
help(plt.scatter)
 
# %%
# use a for loop to check different cluster 
# numbers and see how the inertia changes
inertias = []
k_values = range(1, 10)
for k in k_values: 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(c_num)
    inertias.append(kmeans.inertia_)

#%%
# plot the inertia values to see if there is an elbow in the plot
plt.figure(figsize=(10,5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# %%

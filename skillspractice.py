# %%
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

# %% 
# load the  data, column titles in salary in first row
salary_data = pd.read_csv('2025_salaries.csv', header=1, encoding='latin-1')

stats = pd.read_csv('nba_2025.txt', sep=',', encoding='latin-1')

# %%
# merge the data
merged_data = pd.merge(salary_data, stats, on='Player')
# %%
duplicates = merged_data[merged_data.duplicated(subset='Player', keep=False)]
print(duplicates)
# keep the row with the highest number of games played

# %%
# SK Learn
# 1. create an instance of the model : mymodel = KMeans(n_clusters=3...)
# 2. fit the model to the data: mymodel.fit(X) <-- X is the training dataset
# 3. make predictions using the model: preds = mymodel.predict(X) <-- X is the test dataset
# 4. evaluate performance: score = mymodel.score(X)


# leave salary out of the clusters, see which two features show the clusters the best
# basically use salary as a color variable

# rebounds, minutes played --> good indicators perchance (want to pick features with large variance)
# %%
# Salary is an object column, needs to be numeric
merged_data['Salary'] = merged_data['2025-26'].str.replace('$','').str.replace(',','').astype(float)

# Can't decide whether to use the numeric salary as a 
# colorbar or make it boolean, i tried the boolean and 
# my legend was kind of weird looking? I'll leave the
# code here for when I tried to make it boolean.

# now that it is numeric, we want to make it a boolean, 
# either high or low so we can use it as a color variable
# merged_data['Salary'] = merged_data['Salary'].apply(lambda x: 'High' if x > merged_data['Salary'].median() else 'Low')
# Salary is now a string, either high or low

# this is an example from class that I never finished:
# merged_data['Salary_in_thousands'] = merged_data['Salary'].apply(lambda x: x/1000)

# %%
# drop unnecessary columns
to_drop = ['Awards', 'Player-additional', '2025-26'] 
# awards column has all Nan values, 
# player-additional is not useful,
# 2025-26 can be dropped since we already converted it to
# numeric salary column
merged_data = merged_data.drop(columns=to_drop)
# %%
# fill missing values with 0
# merged_data.isna().sum()
merged_data = merged_data.fillna(0)
#merged_data.isna().sum()

#%%
# drop the duplicate rows, keeping the one with the highest number of games played
merged_data = merged_data.sort_values('G', ascending=False).drop_duplicates(subset='Player', keep='first')
merged_data.shape

# %%
# Run clustering algorithm with best guess for k
from sklearn.cluster import KMeans
# select only numeric features
clust1 = merged_data[['G', 'FG%', '2P%', '3P%', 'TRB', 'MP', 'FT%','ORB','DRB','STL', 'TOV']]
# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
clust1_scaled = scaler.fit_transform(clust1)
# run kmeans
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(clust1_scaled)
# %%
print(kmeans.cluster_centers_)
print(kmeans.labels_)
# %%
# visualize clusters with certain variables in 3d
import cluster_utils as cu
# clusters with field goal percentage, three point percentage, and total rebounds
cu.visualize_clusters_3d(clust1, kmeans.labels_, 'FG%', '3P%', 'TRB', show=True)
# looks relatively flat on the 3 point percentage, there's not
# much variance there so maybe try a different variable?
# %%
# Scale the features of interest and try to visualize in 2d
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = merged_data[['G', 'FG%', '2P%', '3P%', 'TRB', 'MP', 'FT%','ORB','DRB','STL', 'TOV']]
features_scaled = scaler.fit_transform(features)
# %%
# try scatter plots in 2d with salary as color
# field goal percentage and games played
g = plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=merged_data['Salary'], cmap='RdBu')
plt.colorbar(g, label='Salary')
plt.title('Games Played vs Field Goal Percentage')
plt.xlabel('Games Played')
plt.ylabel('Field Goal Percentage')
plt.show()
# %%
# 2 point percentage and total rebounds
g = plt.scatter(features_scaled[:, 2], features_scaled[:, 4], c=merged_data['Salary'], cmap='RdBu')
plt.colorbar(g, label='Salary')
plt.title('2 Point Percentage vs Total Rebounds')
plt.xlabel('2 Point Percentage')
plt.ylabel('Total Rebounds')
plt.show()
# %%
# games played and total rebounds
g = plt.scatter(features_scaled[:, 0], features_scaled[:, 4], c=merged_data['Salary'], cmap='RdBu')
plt.colorbar(g, label='Salary')
plt.title('Games Played vs Total Rebounds')
plt.xlabel('Games Played')
plt.ylabel('Total Rebounds')
plt.show()
# %%
plt.scatter(merged_data['STL'], merged_data['Salary'])
plt.ylabel('Salary')
plt.show()
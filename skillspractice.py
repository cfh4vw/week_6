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
kmeans_sample = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans_sample.fit(clust1_scaled)
# %%
print(kmeans_sample.cluster_centers_)
print(kmeans_sample.labels_)
# %%
# visualize clusters with certain variables in 3d
import cluster_utils as cu
# clusters with field goal percentage, three point percentage, and total rebounds
cu.visualize_clusters_3d(clust1, kmeans_sample.labels_, 'FG%', '3P%', 'TRB', show=True)
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
# turnovers and minutes played
g = plt.scatter(features_scaled[:, 10], features_scaled[:, 5], c=merged_data['Salary'], cmap='RdBu')
plt.colorbar(g, label='Salary')
plt.title('Turnovers vs Minutes played')
plt.xlabel('Turnovers')
plt.ylabel('Minutes Played')
plt.show()
# %%
# minutes played and total rebounds
g = plt.scatter(features_scaled[:, 5], features_scaled[:, 4], c=merged_data['Salary'], cmap='RdBu')
plt.colorbar(g, label='Salary')
plt.title('Minutes Played vs Total Rebounds')
plt.xlabel('Minutes Played')
plt.ylabel('Total Rebounds')
plt.show()
# %%
# look at all scatter plots with salary vs. other features
plt.scatter(merged_data['G'], merged_data['Salary'])
plt.xlabel('Games played')
plt.ylabel('Salary')
plt.title('Games played vs. Salary')
plt.show()
# looks like games played, turnovers, total rebounds,
# minutes played, and steals have high variances, 
# %%
plt.scatter(merged_data['TOV'], merged_data['Salary'])
plt.xlabel('Turnovers')
plt.ylabel('Salary')
plt.title('Turnovers vs. Salary')
plt.show()
# %%
plt.scatter(merged_data['TRB'], merged_data['Salary'])
plt.xlabel('Total rebounds')
plt.ylabel('Salary')
plt.title('Total rebounds vs. Salary')
plt.show()
# %%
plt.scatter(merged_data['MP'], merged_data['Salary'])
plt.xlabel('Minutes played')
plt.ylabel('Salary')
plt.title('Minutes played vs. Salary')
plt.show()
# %%
plt.scatter(merged_data['STL'], merged_data['Salary'])
plt.xlabel('Steals')
plt.ylabel('Salary')
plt.title('Steals vs. Salary')
plt.show()
# %%
# After looking at the 3-var scatter plots with salary
# as the color var, i think I want my clustering
# variables to be minutes played, total rebounds,
# and turnovers

# %%
# try a 3d scatter plot using salary as a color
import plotly.express as px
fig = px.scatter_3d(merged_data, 
                    x='MP', y='TRB', z='TOV',
                    color='Salary')
fig.show()
# this kind of shows a relationship? we want people that 
# have a lower salary, low turnovers, high minutes, and
# high total rebounds
# %%
# Set up kmeans clustering using my features of interest 
cluster = features_scaled[:,[4,5,10]]
# I use features_scaled since I already scaled all the relevant variables.
# run kmeans
kmeans = KMeans(n_clusters=5, random_state=42, verbose=1) 
# using 5 clusters here to see what happens
kmeans.fit(cluster)
# %%
# Evaluate the clustering with total variance explained:
df = pd.DataFrame(cluster, columns=['TRB','MP','TOV'], index=merged_data['Player'])
# df.head() --> looks good

total_sum_squares = np.sum((df - np.mean(df))**2)
total_ss = np.sum(total_sum_squares)
within_ss = kmeans.inertia_
between_ss = total_ss - within_ss

var_ex = between_ss / total_ss
print(f"The total variance explained is {var_ex}.")
# Evaluate the clustering with silhouette scores
# convert cluster to a dataframe, with index as their name for identification

cu.calculate_silhouette_scores(df, random_state=42)
# honestly not great. better than nothing??
# %%
# use elbow method
inertias = []
k_values = range(1, 10)
for k in k_values: 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    inertias.append(kmeans.inertia_)
print(inertias)

# based on the results of the elbow method and the silhouette
# scores, I think that 3 clusters is ideal. As you will see 
# in the plot below, the inertia slows its decrease, and 
# the silhouette value of 3 clusters is relatively high for
# my model at 0.43.

# %%
# visualize the results of the elbow method
plt.figure(figsize=(10,5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()
# %%
# let's re-train the kmeans model with the idea number of clusters (3)
# Set up kmeans clustering using my features of interest 
cluster_fin = features_scaled[:,[4,5,10]]
# I use features_scaled since I already scaled all the relevant variables.
# run kmeans
kmeans_fin = KMeans(n_clusters=3, random_state=42, verbose=1) 
# using 5 clusters here to see what happens
kmeans_fin.fit(cluster_fin)
# %%
# visualize results: 3d scatter plot with salary as the color var,
# and cluster assignment as the shape of the point
df_fin = pd.DataFrame(cluster_fin, columns=['TRB','MP','TOV'], index=merged_data['Player'])
df_fin['cluster'] = kmeans_fin.labels_


fig = px.scatter_3d(df_fin, 
                    x='MP', y='TRB', z='TOV',
                    color=merged_data['Salary'],
                    symbol='cluster',
                    hover_name=df_fin.index)
fig.show()
# add in hover_name so it shows which player is each point
# It is really bothering me how squares and diamonds are used as different
# markers in this plot, but I don't know how/don't think it is possible
# to change it. 
#help(px.scatter_3d)

# %%
# Evaluate performance again: 
# Total variance explained
total_sum_squares = np.sum((df - np.mean(df))**2)
total_ss = np.sum(total_sum_squares)
within_ss = kmeans_fin.inertia_
between_ss = total_ss - within_ss

var_ex = between_ss / total_ss
print(f"The total variance explained is {var_ex}.")

# Silhouette score
cu.calculate_silhouette_scores(df_fin, random_state=42)
# %%

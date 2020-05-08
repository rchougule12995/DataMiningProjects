import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
# Reading the csv file for the first question
df = pd.read_csv("./specs/question_1.csv")

# Using the KMeans function from the sklearn library to create 3 clusters for random state 0
cluster = KMeans(n_clusters=3, random_state=0).fit(df)

# Prediction of the cluster class for the data frame
ClusterClass = pd.DataFrame(cluster.predict(df))

# Assiging the cluster class value to a new column in the dataframe
df['cluster'] = ClusterClass

#print(df.head())

# Generating a scatter plot for the data frame after formation of clusters
fig, ax = plt.subplots()

# Centroid is the variable denoting the cluster center for the data
centroid = cluster.cluster_centers_
ax.scatter(centroid[:, 0], centroid[:, 1], c="red", label ='Class Centroid', linewidth = 6)

# Assiging the x, y values from the dataframe to an array
scatter_x = np.array(df['x'])
scatter_y = np.array(df['y'])

# Plotting each of the point in the data frame according to the cluster value
for g in np.unique(df['cluster']):
    ix = np.where(df['cluster'] == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], label=g, linewidth = 0.1)

# Setting graph/plot labels, title, legend
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

plt.title(label='K Means: 3 clusters', loc='left')
ax.legend(loc='upper left', bbox_to_anchor=(0.4, 1), ncol=2, fancybox=True, shadow=True)
ax.grid(True)
plt.savefig('./output/question_1.pdf')
# Exporting current data frame to the output file
df.to_csv('./output/question_1.csv', index=False)

# ***************************************************************************************
# Reading the question 2 input file
df2 = pd.read_csv('./specs/question_2.csv')

# print(df2.head())
# Discarding the irrelvant columns in clustering the dataframe
df2_new = df2.drop(['NAME', 'MANUF', 'TYPE', 'RATING'], axis=1)

#print(df2_new.head())
# Initializing the kmeans cluster with 5 clusters and 0 random state for the dataframe, with 5 maximum runs
# 100 maximum iterations
kmeans2 = KMeans(n_clusters=5, random_state=0, n_init=5, max_iter=100).fit(df2_new)

# Predicting the cluster class from the kmeans
ClusterClass2 = pd.DataFrame(kmeans2.predict(df2_new))
print(kmeans2.inertia_)
# Assign the cluster class value to the new attribute in the data frame
df2_new['config1'] = ClusterClass2

# Using the kmeans with 5 clusters, and 0 random state 100 maximum runs and 100 optimization steps
kmeans2 = KMeans(n_clusters=5, random_state=0, n_init=100, max_iter=100).fit(df2_new)
print(kmeans2.inertia_)
# Predicting the cluster class from the kmeans
ClusterClass2 = pd.DataFrame(kmeans2.predict(df2_new))

# Assigning the cluster class to a new attribute in the dataframe
df2_new['config2'] = ClusterClass2

# Using the kmeans with 3 clusters, and 0 random state, 5 maximum runs and 100 optimization steps
kmeans2 = KMeans(n_clusters=3, random_state=0, n_init=5, max_iter=100).fit(df2_new)
print(kmeans2.inertia_)
# Predicting the cluster class from the kmeans
ClusterClass2 = pd.DataFrame(kmeans2.predict(df2_new))

# Assigning the cluster class to a new attribute in the dataframe
df2_new['config3'] = ClusterClass2

# Exporting the dataframe to csv
df2_new.to_csv('./output/question_2.csv', index=False)

# ***********************************************************************
# Importing the csv input file
df3 = pd.read_csv("./specs/question_3.csv" )

# Dropping the ID attribute
df3_new = df3.drop(['ID'], axis=1)
# print(df3_new.head())

# Using the kmeans with 7 clusters, and 0 random state, 5 maximum runs and 100 optimization steps
kmeans3 = KMeans(n_clusters=7, random_state=0, n_init=5, max_iter=100).fit(df3_new)

# Predicting the cluster class from the kmeans
ClusterClass3 = pd.DataFrame(kmeans3.predict(df3_new))

# Adding the kmeanse cluster class value to a new variable
df3_new['kmeans'] = ClusterClass3

# Plotting the dataframe values for clustering in scatter plot
fig, ax = plt.subplots()

centroid3 = kmeans3.cluster_centers_
ax.scatter(centroid3[:, 0], centroid3[:, 1], c="red", label ='Class Centroid', linewidth = 6)

scatter_x = np.array(df3_new['x'])
scatter_y = np.array(df3_new['y'])

for g in np.unique(df3_new['kmeans']):
    ix = np.where(df3_new['kmeans'] == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], label=g, linewidth = 0.1)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

plt.title(label='K Means: 7 clusters', loc='left')
ax.legend(loc='center', bbox_to_anchor=(0.6, 1.06), ncol=3, fancybox=True, shadow=True)
ax.grid(True)
plt.savefig('./output/question_3_1.pdf')
# Keeping a copy of the original data frame and working on the copy
df4 = df3_new

# Dropping the K means column in the copy data frame as it will not be required in the DBScan
df4 = df4.drop(['kmeans'],axis=1)

# Normalizing the x, y column in the data frame
df4['x'] = ((df4['x']-df4['x'].min())/(df4['x'].max()-df4['x'].min()))
df4['y'] = ((df4['y']-df4['y'].min())/(df4['y'].max()-df4['y'].min()))

# Using the DBSCAN method for clustering as the data is highly dense.
clustering_df3 = DBSCAN(eps=0.04, min_samples=4).fit(df4)
clustering_df4 = DBSCAN(eps=0.08, min_samples=4).fit(df4)

print(metrics.silhouette_score(df4, clustering_df3.labels_))
print(metrics.silhouette_score(df4, clustering_df4.labels_))

# Assiging the Cluster labels using epsilon 0.04 to dbscan1
df4['dbscan1'] = clustering_df3.labels_

# Assiging the Cluster labels using epsilon 0.08 to dbscan2
df4['dbscan2'] = clustering_df4.labels_

df4.insert(2, 'kmeans', df3_new['kmeans'])

# Exporting dataset to csv
df4.to_csv('./output/question_3.csv', index=False)

# *****************************************************
# Plotting the cluster values from DBScan1 and DBScan2 values
fig, ax = plt.subplots()
scatter_x = np.array(df4['x'])
scatter_y = np.array(df4['y'])

for g in np.unique(df4['dbscan1']):
    ix = np.where(df4['dbscan1'] == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], label=g, linewidth = 0.01)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

plt.title(label='Clustering DBScan: epsilon 0.04', loc='left')
ax.legend(loc='center left', bbox_to_anchor=(0.6, 1.06), ncol=3, fancybox=True, shadow=True)
ax.grid(True)
plt.savefig('./output/question_3_2.pdf')
# Exporting the graph

fig, ax = plt.subplots()
scatter_x = np.array(df4['x'])
scatter_y = np.array(df4['y'])

for g in np.unique(df4['dbscan2']):
    ix = np.where(df4['dbscan2'] == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], label=g, linewidth = 0.01)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

plt.title(label='Clustering DBScan: epsilon 0.08', loc='left')
ax.legend(loc='center left', bbox_to_anchor=(0.6, 1.06), ncol=3, fancybox=True, shadow=True)
ax.grid(True)
plt.savefig('./output/question_3_3.pdf')

# print(df4)


# Project2_Wheat_data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Edureka_Notebook\\Certification_project\\Project2\\Wheat_data.csv')
df.head()

df.shape

# Change the headers to country and year accordingly.

df.columns = ['Country','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007']
df.head()

df1= df.iloc[:,1:]

# Cleanse the data if required and remove null or blank values

df1.fillna(0)
for i in df1.columns:
    df1[i] = df1[i].astype(str)
    df1[i] = df1[i].str.replace(',','')
    #df1[i] = pd.to_numeric(df[i])
    
 # Apply PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df1)
X1=pca.transform(df1)
X1 = pd.DataFrame(X1)
X1.columns = ['PCA1','PCA2']
X1.head()

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

# Plot elbow chart or scree plot to find out optimal number of clusters.

from sklearn.cluster import KMeans

k1=[]
for k in range(2,20):
    kmeans = KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(X1)
    k1.append(kmeans.inertia_)

plt.plot(range(2,20),k1)
plt.title('Elbow Method')
plt.xlabel('Value of K')

kmeans = KMeans(n_clusters=7)
kmeans.fit(X1)
print(kmeans.cluster_centers_)

print(kmeans.labels_)
df['Cluster_No'] = kmeans.labels_
df.head()

# Plot the cluster data
sns.lmplot('PCA1','PCA2',data=X1,hue='Cluster_No',
            height=6, aspect=1, fit_reg=False)
            
# You can either choose to group the countries based on years of data or using the principal components

X1.sort_values(['Cluster_No','PCA1','PCA2'])
df.sort_values(['Cluster_No'])

# save file
X1.to_csv('output.csv',index=True)

df2 = df.set_index('Country')
df2.drop(['Cluster_No'],axis=1)

# Then see which countries are consistent and which are largest importers of the good based on scale and position of cluster.

# Largest Importer and constantly increasing
df2.loc["Sierra Leone"]
X = df2.loc["Sierra Leone"].index[0:18]
Y = df2.loc["Sierra Leone"].values[0:18]

plt.bar(X, Y)
plt.setp(plt.gca().get_xticklabels(), rotation=90,
         horizontalalignment='right')  # Rotate Axis Labels

plt.show()

# most consistent

x = df2.loc['Monaco'].index[0:18]
y = df2.loc['Monaco'].values[0:18]

plt.bar(x, y)
plt.setp(plt.gca().get_xticklabels(), rotation=90,
         horizontalalignment='right')  # Rotate Axis Labels




#%% [markdown]
# ## Task 2
# Author: Sam Higgs
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'A4'))
	print(os.getcwd())
except:
	pass

# Preferred data wrangling tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Use for evaluation
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as arscore
from sklearn.metrics.pairwise import euclidean_distances 

# Clustering and wrangling algorithms used
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from seaborn import heatmap

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
rcParams['figure.figsize'] = 6,6

# HELPER FUNCTIONS
def get_strong_corr(cor, cutoff=0.8):
    np.fill_diagonal(cor.values, np.NaN)
    high_cor = cor.where((cor >= cutoff) if cutoff > 0 else (cor <= cutoff))
    high_cor = high_cor.dropna(how='all')
    high_cor = high_cor.dropna(axis='columns', how='all')
    return high_cor

def teach_kmeans(data, nClust=3):
    model = KMeans(nClust)
    model.fit(data)
    clust_labels = model.predict(data)
    cent = model.cluster_centers_
    return clust_labels, cent

def reduce_data(X, dim=2):
    pca = PCA(n_components=dim, whiten=False)
    reduced = pca.fit_transform(X)
    return pd.DataFrame(reduced, columns=['x','y'], index=None)

def plot_clusters(x, y, label, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x,y,c=label, s=50)
    ax.set_title(title)
    plt.colorbar(scatter)
    plt.plot()

def plot_all(x, y, labels, title):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8,5))
    ax1.scatter(x, y, c=labels[0], s=50)
    ax2.scatter(x, y, c=labels[1], s=50)
    ax3.scatter(x, y, c=labels[2], s=50)
    ax1.set_title('Kmeans')
    ax2.set_title('dbscan')
    ax3.set_title('agcluster')
    plt.plot()
#% [markdown]
#       ### (a) Import and plot the data and if necessary preprocess it using sklearn.preprocessing.
#       ### (a) *ANSWER
#%% 
# Importing the data
assert os.path.isfile('data\\wine.data')
assert os.path.isfile('data\\wine.names')

df_names = [
    'label',
    'Alcohol',
 	'Malic acid',
 	'Ash',
	'Alcalinity of ash  ',
 	'Magnesium',
	'Total phenols',
 	'Flavanoids',
 	'Nonflavanoid phenols',
 	'Proanthocyanins',
	'Color intensity',
 	'Hue',
 	'OD280/OD315 of diluted wines',
 	'Proline'
     ]

df = pd.read_csv('data\\wine.data', header=None, names=df_names, index_col=False, dtype=np.float)

# How I decided what column to drop
df.head()
df.describe()
# Plot
cor = df.iloc[:,1:].corr()
heatmap(cor, square=True)
pve_cor = get_strong_corr(cor, cutoff=0.8)
neg_cor = get_strong_corr(cor, cutoff=-0.8)

df.columns
df.drop(columns='Total phenols', inplace=True)
# No highly correlated negatives.

sc = StandardScaler(copy=False, with_mean=True, with_std=True)
transformed_df = pd.DataFrame(sc.fit_transform(df.iloc[:,1:]), columns = df.columns[1:], index=None)
transformed_df['label'] = df['label']
transformed_df
#%% [markdown]
#       ### (b) Try three different clustering techniques already implemented in scikit-learn. Please state which algorithms
#               you have used and what parameters you have chosen and why you have chosen them.Method 1 : KMeans
#       ### (b) *ANSWER*
#              NOTE: I would normally use a training validation and test set for this, however as this was not the purpose of the task
#               and I didn't want to run out of time I chose to train the model on the full dataset so the accuracy may be slightly off.
#               * KMeans, 
#                   ** n_clusters=3, as there were 3 classes
#               * DBSCAN
#                   ** eps=2.1
#               * Agglomorative Clustering
#

#%% [markdown]
#       #### Method 1 KMEANS
#%%
reduced_data = reduce_data(transformed_df.iloc[:,:-1]) 
reduced_data['label'] = df['label']
kmeans_label, cent = teach_kmeans(transformed_df.iloc[:,:-1])
kmeans_label = [c+1 for c in kmeans_label]
reduced_data['kmeans_label'] = kmeans_label
# plot_clusters(reduced_data['x'], reduced_data['y'], reduced_data['kmeans_label'], 'KMeans Clustering')
# DBSCAN
dbscan = DBSCAN(eps=2.1).fit(transformed_df.iloc[:,1:])
dbs_labels = [l+2 for l in dbscan.labels_]
reduced_data['dbscan_label'] = dbs_labels
# plot_clusters(reduced_data['x'], reduced_data['y'], reduced_data['dbscan_label'], 'DBSCAN Clustering')
# AGGLOMERATIVE CLUSTERING
agc = AgglomerativeClustering(n_clusters=3, affinity='euclidean').fit(transformed_df.iloc[:,:-1])
reduced_data['agc_label'] = agc.labels_
# plot_clusters(reduced_data['x'], reduced_data['y'], reduced_data['agc_label'], 'Agglomorative Clustering')
plot_all(reduced_data['x'], reduced_data['y'], [reduced_data['kmeans_label'], reduced_data['dbscan_label'],reduced_data['agc_label']], 'Agglomorative Clustering')
#%% [markdown]
# ## Evaluation

scores = []
scores.append(['KMEANS', 
    nmi(reduced_data['label'], reduced_data['kmeans_label'], average_method='arithmetic'),
    arscore(reduced_data['label'], reduced_data['kmeans_label'])])

scores.append(['DBSCAN', 
    nmi(reduced_data['label'], reduced_data['dbscan_label'], average_method='arithmetic'), 
    arscore(reduced_data['label'], reduced_data['dbscan_label'])])

scores.append(['AGGLOMORATIVE', 
    nmi(reduced_data['label'], reduced_data['agc_label'], average_method='arithmetic'), 
    arscore(reduced_data['label'], reduced_data['agc_label'])])

scores = sorted(scores, key=lambda x: x[1], reverse=True)
print(f"{'Algorithm':^15} {'NMI Score':^15} {'Adjusted RAND Score':^15}")
for s in scores:
    # s_f = [float(sf) for sf in s[1:]]
    # print(type(s[0]))
    print(f'{s[0]:>15} {s[1]:^15.3f} {s[2]:^15.3f}')





#%%

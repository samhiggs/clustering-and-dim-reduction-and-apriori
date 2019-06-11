#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'A4'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Assignment 4 Foundation of Data Analysis
# ## Author: Sam Higgs
# ### 1) SVD Image Compression Exercise
# Singular Vector Decomposition (SVD) is a popular tool for dimensionality reduction.  
# The purpose of SVD is to break down a matrix into simpler components, making the calculations less costly.  
# Given a matrix M x N, the simpler components are a U (m x m) matrix, a sigma (m x n) and a transposed V which is  
# the transposition of (n x n).  
#   
# In the context of image compression, we want to reduce the dimensionality of our data, without effecting the  
# images dimensions.  
# We can display images as a matrix of values, where each pixel represents the light intensity (as we are using greyscale)  
# The decomposition into U, sigma and VT enables us to approximate the original image, whilst using far less  
# memory, whilst still giving us an accurate representation of the image.

#%%
import matplotlib.image as npim
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from numpy import zeros, dot, diag, array
from numpy.linalg import svd, eigh
from ipywidgets import interact, fixed

from skimage.measure import compare_ssim

#%%
# Increase default figure-plotting quality
img = npim.imread("faculty.png")
mpl.rcParams['figure.dpi'] = 300
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#%%
def measure_sim(imgA, imgB):
    (sim_score, diff) = compare_ssim(imgA, imgB, full=True)
    sim_score = (sim_score * 255).astype("uint8")
    return sim_score

def compress_and_show(img, compression):
    """
    Compress the greyscale image and display the plot.
    """
    recon_img = svd_compression(img, compression)
    sim_score = measure_sim(img, recon_img)

    print(f"Image similarity score: {sim_score:.2f}")

    f, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(8,5))
    ax1.set_title("Original Image")
    ax1.imshow(img, cmap = 'gray')
    ax2.set_title("Compressed Image. Ratio:{:.2f}%".format(100*compression))
    ax2.imshow(recon_img, cmap = 'gray')
    ax2.axis('off')
    f.tight_layout()
    return sim_score

def svd_compression(img, compression):
    comp_ratio = int(compression*img.shape[0])
    U,s,VT = svd(img)
    return dot(U[:,:comp_ratio], dot(diag(s[:comp_ratio]), VT[:comp_ratio,:]))

def print_corr(corr, labels):
    print('\n\n')
    plt.figure(figsize=(8,8))
    plt.matshow(corr, fignum=1)
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.plot()
#%% [markdown]
# Below, you can move the slider between 0 and 1 (the compression ratio) to see the impact of  
# of a large or small number of components. When the compression ratio is close to 1, the image is  
# closest to the original depiction and has minimal to no compression.
# 
#%%
interact(compress_and_show, compression=(0.00,1.00,0.02), img=fixed(img), continuous_update=False)
#%% [markdown]
# Below you can see the trend as you increase the compression. There is a logarithmic shape. It is therefore
# best to have a compression > 10-15% to get the most "value" out of your image. Before then, the quality is
# substantially worse, after then there is a greater tradeoff.
#%%
sim_scores = []
for i in range(0,100,2):
    sim_scores.append([i, measure_sim(img, svd_compression(img, i/100))])
sim_scores = array(sim_scores)
plt.plot(sim_scores[:,0], sim_scores[:,1])
plt.title('Similarity Score ratio at different compression')
plt.xlabel('Compression')
plt.ylabel('Similarity Score Ratio')
#%% [markdown]
# ### 2) Comparing PCA vs. SVD
# ### 2) a.
if not os.path.isfile('data\\BreastTissue.csv'):
    assert "Data file does not exist"

dtypes = { 
    'names': 
    ('Case #', 'Class', 'I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP',
       'DR', 'P'),
    'formats': (np.int32, np.object, np.float, np.float, np.float, np.float, 
        np.float, np.float, np.float, np.float, np.float)
}
data = np.loadtxt(fname='data\\BreastTissue.csv', dtype=dtypes, delimiter=',', skiprows=1, unpack=True)
#%%
#Represent categories of classes as values between 0 and 1, evenly distributed
enum_class = np.array(np.unique(data[1], return_inverse=True)[1], np.float)
enum_class /= enum_class.max()
data[1] = enum_class

#%% [markdown]
# ### 2) b.
# We can see from the below correlation matrix, which represents the strongest correlation as yellow, and
# the strongest negative correlation as dark navy blue.  
# The noticable correlations are printed below. It is worth nothing that there are limited classes, so there is 
# a higher chance that they are correlated.

#%%
#Convert data to numpy array and calculate the correlation coefficients
npData = np.array(data)
npData.shape
corr = np.corrcoef(npData[1:])

#find the highly correlated values and print them.
d_names = dtypes['names'][1:]
corrI = np.where(corr>=0.8)
corrI = set(zip(corrI[0], corrI[1]))
corrI = [(d_names[coord[0]], d_names[coord[1]]) for coord in corrI if coord[0] != coord[1]]
[print(f"{c[0]:<10} and {c[1]:>10}") for c in corrI]


#Remove duplicates as there will be 2 of each
cleaned_corr = []
for i, c in enumerate(corrI):
    flip = (c[1], c[0])
    if flip not in corrI[i:]:
        cleaned_corr.append(c)

#%% [markdown]
# #### Drop correlated variables
# We now need to drop highly related variables. We can see that IO has a strong correlation
# Unnecessary as there are no strong negative correlations
# negCorrI = np.where(corr<=-0.8)
# negCorrI = list(zip(negCorrI[0], negCorrI[1]))
print("Here are the strongly correlated values in the data")
[print(f"{c[0]:<10} and {c[1]:>10}") for c in cleaned_corr]
print_corr(corr, dtypes['names'][1:])
to_drop = ['Case #', 'DA', 'P', 'Max IP', 'Area']
to_drop_idx = sorted([dtypes['names'].index(e) for e in to_drop])
indData = np.delete(npData, to_drop_idx, 0)
indDataNames = [n for n in dtypes['names'] if n not in to_drop]
assert indData.shape[0] == npData.shape[0] - len(to_drop)

# with DA, P and Max IP so we can easily reduce the dimensionality but dropping DA, P and Max IP
ind_corr = np.corrcoef(indData)
print_corr(ind_corr, indDataNames)
axis = 1
indData_scaled = indData / np.expand_dims(np.std(indData, axis=axis), axis)
std_corr = np.corrcoef(indData_scaled)
print_corr(std_corr, indDataNames)
# standardised = indData - np.expand_dims(np.mean(indData, axis=axis), axis)
#%%[markdown]
# ##Report on original correlation vs unit-variance scaled 
#%%
eigenVals, vectors = eigh(ind_corr)
classes = [x for x in sorted(zip(dtypes['names'][1:], eigenVals, vectors), key=lambda x:x[1], reverse=True)]
std_eigenVals,std_vectors = eigh(std_corr)
std_classes = [x for x in sorted(zip(indDataNames, std_eigenVals, std_vectors), key=lambda x:x[1], reverse=True)]

print(f"""We can see from the eigenvalues the principal components in order:
    """)
# print(f"{'Feature':>5} {'Eigenvalue':^15} {'Variance (%)':^15}")
# [print(f"{c[0]:>5} {c[1]:^15.4f} {c[1]/sum(eigenVals):^15.2f}") for i,c in enumerate(classes)]
print(f"{'Standardised Data':^17}")
print(f"{'Feature':>5} {'Eigenvalue':^15} {'Std Eigenvalue':^15} {'Variance (%)':^15}")
[print(f"{c[0]:>5} {classes[i][1]:^15.4f} {c[1]:^15.4f} {c[1]/sum(std_eigenVals):^15.2f}") for i,c in enumerate(std_classes)]

#%% [markdown]
# ##TODO
# * [X] Remove attributes that are not independent
# * [ ] Describe what you would do if you had a mix of numerical and categorical variables
# * [ ] Calculate the correlation matrix and predict what the principal components might be
# * [ ] Scale data to the unit variance but DO NOT centre it.
# * [ ] Conduct PCA on original data
# * [ ] Conduct PCA on scaled data
# * [ ] Plot the eigenvalues and describe how much variance of the original features is explained by each PC
# * [ ] Explain why it would be sensible to scale the data before applying PCA. Include problems that might occur if it is not scaled
# * [ ] Centre the data and conduct a SVD on the standardised data.
# * [ ] Calculate the eigenvalues from the SVD above and explain the connection that is evident when comparing that to the previous PCA results.

#%%
# Hint: Useful for 2c & d)
from sklearn.preprocessing import StandardScaler




#%%

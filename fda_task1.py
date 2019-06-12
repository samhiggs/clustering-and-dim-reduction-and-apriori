#%% 
# Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'A4'))
	print(os.getcwd())
except:
	pass

import matplotlib.image as npim
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from numpy import zeros, dot, diag, array
from numpy.linalg import svd, eigh
from ipywidgets import interact, fixed

from skimage.measure import compare_ssim
from sklearn.preprocessing import StandardScaler

# HELPER FUNCTIONS
def measure_sim(imgA, imgB):
    (sim_score, diff) = compare_ssim(imgA, imgB, full=True)
    sim_score = (sim_score*100).astype("uint8")
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
    f.tight_layout()
    return sim_score

def svd_compression(img, compression, return_components=False):
    comp_ratio = int(compression*img.shape[0])
    U,s,VT = svd(img)
    if not return_components:
        return dot(U[:,:comp_ratio], dot(diag(s[:comp_ratio]), VT[:comp_ratio,:]))
    else:
        return U, s, VT, dot(U[:,:comp_ratio], dot(diag(s[:comp_ratio]), VT[:comp_ratio,:]))


def print_corr(corr, std_corr, labels):
    print('\n\n')
    f, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(5,5))
    ax1.set_title('Original dataset')
    ax1.matshow(corr)
    ax2.set_title('Standardised dataset')
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    ax2.matshow(std_corr)
    plt.show()
#%% [markdown]
# # Assignment 4 Foundation of Data Analysis
# ## Author: Sam Higgs
# ## Task 1 : Dimensionality Reduction, SVD and PCA (30 pts)
#   ### Part A) SVD Image Compression Exercise

#       #### (a) Conduct a SVD using numpy linalg svg. 
#           Retrieve the left and right singular vectors. 
#           Reconstruct and plot the original image from the computed factorization
#       #### (a) ANSWER:
#          Below you can see the trend as you increase the compression. There is a logarithmic shape. It is therefore
#          best to have a compression > 10-15% to get the most "value" out of your image. Before then, the quality is
#          substantially worse, after then there is a greater tradeoff.

# Increase default figure-plotting quality
img = npim.imread("faculty.png")
mpl.rcParams['figure.dpi'] = 300
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
# compress_and_show(img, .2)
# Interactively run the compression to see the difference.
interact(compress_and_show, compression=(0.00,1.00,0.1), img=fixed(img), continuous_update=False)

#%% [markdown]
#        #### (b) Describe the properties of the individual components i.e. inspect the properties of the vectors/matrices
#           including the original matrix.
#           Singular Vector Decomposition (SVD) is a popular tool for dimensionality reduction.  
#           The purpose of SVD is to break down a matrix into simpler components, making the calculations less costly.  
#           Given a matrix M x N, the simpler components are a U (m x m) matrix, a sigma (m x n) and a transposed V which is  
#           the transposition of (n x n).  

U, s, VT, recon_img = svd_compression(img, .1, True)
print(f"""Individual components are broken down into 3 matrices for the purpose of 
simplifying the components and therefore reducing the computational complexity.
The SVD theorem states that a matrix M = U*sigma*V^T. We can see the properties 
of each component below.
The non-zero singular values are in decreasing order to ensure that we are reducing the least important 
values, that way when it's reconstructed, the image has the minimal loss possible.

    Image : {img.shape}
    left-singular : {U.shape}
    non-zero singular : {s.shape}
    right-singular : {VT.shape}
    Reconstructed Image : {recon_img.shape}

The image's vector looks like:
{img.view()}


The left-singular vector looks like:
{U.view()}


The non-zero singular values are positive, stored in decreasing order. 
They represent the eigenvalues of the non-zero diagonals.
{s.view()}


The right-singular vector looks like:
{VT.view()}

The reconstructed image vector looks like:
{recon_img.view()}

""")
#%% [markdown]
#       #### (c) Inspect the vector containing the singular values. The original matrix can be compressed using a number
#           of singular values that you choose. What happens to your matrix A when you choose a very high or very
#           low number of singular values? If you intended to reduce the dimensionality of your data, what would be
#           the number of singular values you would choose and why?
#       
#       ####  (c) *ANSWER*:
#           When you change the number of singular values (always less than the original image number). You will remove some
#           of images data in the form of pixel density. The result is that more pixels will appear the same in an image. 
#           Below shows a similarity score at different compressions (i.e with a different number of singular values). 
#           This provides a good indicator of where the compression delivers the most value. You can see by the logarithmic shape, that when there is high compression, the images are less similar.
#           However the similarity increases dramatically by increasing the quality by a small amount.
#           I would therefore use a compression of around 80%. 
#           Get a range of different measures to see how the compression effects the image quality.
sim_scores = []
for i in range(0,100,10):
    sim_scores.append([100-i, measure_sim(img, svd_compression(img, i/100))])
sim_scores = array(sim_scores)
plt.plot(sim_scores[:,0],sim_scores[:,1])
plt.title('Similarity Score ratio at different compressions')
plt.xlabel('Compression')
plt.ylabel('Similarity Score Ratio')

#%% [markdown]
#   ### Part B) Comparing PCA to SVD (20 points)
#       ### (a) Convert the dataset into a text file (e.g. .csv) and load it using numpy.loadtxt. Do not use attributes that
#               cannot be considered independent variables. What would you do if you had a mix of numerical and
#               categorical variables?
        ### (a) *ANSWER*
#               We can see from the below correlation matrix, which represents the strongest correlation as yellow, and
#               the strongest negative correlation as dark navy blue.  
#               The noticable correlations are printed below. It is worth nothing that there are limited classes, so there is 
#               a higher chance that they are correlated.

#%%
if not os.path.isfile('data\\BreastTissue.csv'): assert "Data file does not exist"
dtypes = { 'names': ('Case #', 'Class', 'I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP', 'DR', 'P'),
            'formats': (np.int32, np.object, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float)
        }
data = np.loadtxt(fname='data\\BreastTissue.csv', dtype=dtypes, delimiter=',', skiprows=1, unpack=True)
#Represent categories of classes as values between 0 and 1
enum_class = np.array(np.unique(data[1], return_inverse=True)[1], np.float)
enum_class /= enum_class.max()
data[1] = enum_class
#Convert data to numpy array and calculate the correlation coefficients
npData = np.array(data)
corr = np.corrcoef(npData[1:])
#find the highly correlated values and print them.
d_names = dtypes['names'][1:]
corrI = np.where(corr>=0.8)
corrI = set(zip(corrI[0], corrI[1]))
corrI = [(d_names[coord[0]], d_names[coord[1]]) for coord in corrI if coord[0] != coord[1]]
# Remove duplicates as there will be 2 of each
cleaned_corr = []
for i, c in enumerate(corrI):
    flip = (c[1], c[0])
    if flip not in corrI[i:]:
        cleaned_corr.append(c)
# Drop highly correlated values
print("Here are the strongly correlated values in the data")
[print(f"{c[0]:<10} and {c[1]:>10}") for c in cleaned_corr]

to_drop = ['Case #', 'DA', 'P', 'Max IP', 'Area']
to_drop_idx = sorted([dtypes['names'].index(e) for e in to_drop])
indData = np.delete(npData, to_drop_idx, 0)
indDataNames = [n for n in dtypes['names'] if n not in to_drop]
assert indData.shape[0] == npData.shape[0] - len(to_drop)
ind_corr = np.corrcoef(indData)
plt.figure(figsize=(2,2))
plt.xticks([0,1,2,3,4,5], indDataNames)
plt.yticks([0,1,2,3,4,5], indDataNames)
plt.matshow(ind_corr, fignum=1)
plt.show()
#%% [markdown]
#           #### (b) Find the correlation matrix using np.corrcoef, inspect it and try to predict which variables will form
#                   principal components. This will also help you in the next step. There are no right answers because it is a
#                   complex problem, which is why we use PCA. However, you will be able to predict some trends that will
#                   be reflected in the PCA.
#           #### (b) *ANSWER*
#                   Classes will likely have a high PCA score as there are few classes. The values with less +- correlation will 
#                   likely have less impact.

#%% [markdown]
#           #### (c) Scale your data to unit variance but do not center it. 
#                    Conduct a PCA through calculating the eigenvalues
#                    and vectors of the covariance matrix of the scaled data as well as of the original data using the
#                    numpy.linalg.eigh command. 
# 
#                    Elaborate your answer with a plot and explain how much variance of the original features is explained by the PCs, by
#                    computing the variance that is explained by each PC. Why would it be sensible to scale the data in our
#                    case before applying PCA and what are problems arising when conducting PCA on the unscaled data?
#           #### (c) *ANSWER*
#                    *How many components would you choose for either PCA?* 
#                        You can see that across the original, scaled and centered data, the DR feature is strongest. There is a difference between the
#                        rate and therefore the proportion that they are represented by.     
#                        For the original and scaled dataset I would choose HFS, A/DA and DR. Whilst, for the centered data I would consider DR and A/DA.
#                   *how much variance of the original features is explained by the PCs, by computing the variance that is explained by each PC.*
#                        You can explain all the variance by the principal components...
#                   * Why would it be sensible to scale the data in our case before applying PCA and what are problems arising when conducting PCA on the unscaled data?*
#                        Scaling the data reduces the chance of data over-representing certain features.

#%%
# Scale to unit-variance (stdandard deviation) and scaled to both
std_data = StandardScaler(with_mean=False).fit_transform(indData)
std_corr = np.corrcoef(std_data)
centered_data = StandardScaler().fit_transform(indData)
centered_corr = np.corrcoef(centered_data)

# print_corr(ind_corr, std_corr, indDataNames)
eigenVals, vectors = eigh(ind_corr)
std_eigenVals, std_vectors = eigh(std_corr)
centered_eigenVals, centered_vectors = eigh(centered_corr)

classes = [x for x in sorted(zip(indDataNames, eigenVals, vectors), key=lambda x:x[1], reverse=True)]
std_classes = [x for x in sorted(zip(indDataNames, std_eigenVals, std_vectors), key=lambda x:x[1], reverse=True)]
centered_classes = [x for x in sorted(zip(indDataNames, centered_eigenVals, centered_vectors), key=lambda x:x[1], reverse=True)]

print(f"{'Feature':>5} {'Eigenvalue':^15} {'Std Eigenvalue':^15} {'Ctred Eigenvalue':^15} {'Variance (%)':^15} {'std Variance (%)':^15} {' ctred Variance (%)':^15}")
s = [print(f"{c[0]:>5} {classes[i][1]:^15.4f} {c[1]:^15.4f} {centered_classes[i][1]:^15.4f} {classes[i][1]/sum(eigenVals):^15.2f} {c[1]/sum(std_eigenVals):^15.2f} {centered_classes[i][1]/sum(centered_eigenVals) :^15.2f}") for i,c in enumerate(std_classes)]
plt.plot(indDataNames, eigenVals/sum(eigenVals), indDataNames, std_eigenVals/sum(std_eigenVals), indDataNames, centered_eigenVals/sum(centered_eigenVals))
plt.legend(labels=('Original', 'Scaled', 'Centered'))
plt.show()

#%% [markdown]
#           #### (d) Now additionally center your data and conduct a SVD on the standardized data. Use it to retrieve the
#                   eigenvalues of the covariance matrix. What connection between SVD and PCA can you observe?
#           #### (d) *ANSWER*
#                   My results for the eigenvalues are exactly the same when both are centered.
#
#%%
U,s,VT = svd(centered_corr)
recon_data = dot(U, dot(s, VT))
print(s.view())
print('\n')
print(np.flip(centered_eigenVals))

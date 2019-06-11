#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'A4'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Task 3 - Apriori Algorithm
# Author: Sam Higgs
# GOAL: Implement the Apriori algorithm and apply it to mine frequent itemsets for movie recommendations.
# ### TODO:
# * [ ] import dataset with pandas
# * [ ] Count the support of each item, storing it in a table
# * [ ] map movies to indices
# * [ ] Display some high level data on the dataset including the highest to lowest movie frequency
# * [ ] Export length-1 frequent movies with their absolute supports into a text file
# * [ ] Use the apriori model to recommend movies based on previous recommendations
# * [ ] Explain the choice
# * [ ] 

# ### Definitions:
# * Apriori : 
# * length-1 frequency : 
# * itemset : An itemset is the set of movie recommendations a user has given 5 stars. Each row is an itemset
# * absolute support : 
# * support threshold : The support threshold is the minimum frequency an item must appear in the database to be considered.
# * candidate : 
# * frequent itemsets : All the sets which contain the item with the minimum support
# * Apriori property : Any subset of frequent itemset must be frequent 
# * frequent itemsets : 


#%%
# Use for evaluation
#%% [markdown]
# ### Import Packages
import csv
import numpy as np
import matplotlib as plt
import itertools
from itertools import combinations
import pydot
from graphviz import Digraph
import IPython.display as ipd

from HashTree import HNode, HTree

#%% [markdown]
# ### Import Dataset
def import_data(fn):
    assert os.path.isfile(fn)
    itemset = np.loadtxt(fn, dtype=np.object, delimiter='\n')
    itemset = [set(m.split(';')) for m in itemset]
    hashed_data = {}
    for s in itemset:
        for movie in s:
            if movie in hashed_data:
                hashed_data[movie] +=1
            else:
                hashed_data[movie] = 1
    encodings = {}
    encodings_reverse = {}
    for i, k in enumerate(hashed_data.keys()):
        encodings[k] = i
        encodings_reverse[i] = k
    return itemset, hashed_data, encodings, encodings_reverse

def output_support(hashed_data):
    fn = 'oneItems.txt'
    with open(fn, 'w') as f:
        for k,v in hashed_data.items():
            f.write(str(v) + ':' + k + '\n')

def generate_patterns(data, hashedItems, min_support=0.05):
    support = int(min_support/100.0*len(data))
    freq_itemset = {}
    freq_itemset[1] = hashedItems
    pattern_helper(data, hashedItems, support, 2, freq_itemset)
    return freq_itemset

def pattern_helper(data, itemset, support, level, freq_itemset):
    split_data = []
    print(data[:10])
    # for movie in data:
    #     for m in movie:
    #         print(m)
    if len(itemset) == 0:
        return
    
def encode_dataset(itemset, encodings):
    encoded_itemset = []
    for i in itemset:
        e = set()
        for m in i:
            e.add(encodings[m])
        encoded_itemset.append(e)
    return encoded_itemset

def generate_graph(data):
    graph = pydot.Dot(graph_type='graph')
    for k,v in data.items():
        edge = pydot.Edge('\{\}', f'{list(k)[0]}\n{v}')
        graph.add_edge(edge)
    return graph

def prune_itemset(hashmap, itemset):
    size = len(list(hashmap.keys())[0])
    prune_itemset = {e for k in hashmap.keys() for e in k}
    print(prune_itemset)
    return [[e for e in i if e in prune_itemset] for i in itemset]
    

def add_to_graph(graph, data):
    for k,v in data.items():
        parent = list(k)[0]
        print(parent)

def shingle_hash(itemset, k, support=500):
    k_freq_itemset = []
    k_freq_hashed = {}
    to_keep = set()
    for i in itemset:
        if len(i) < k:
            continue
        comb = set(combinations(i,k))
        comb = set(frozenset(sorted(c)) for c in comb)
        for c in comb:
            key = c
            if key not in k_freq_hashed:
                k_freq_hashed[key] = 1
            el  se:
                k_freq_hashed[key] += 1
            if key not in to_keep and k_freq_hashed[key] > support:
                to_keep.add(key)
    discard = set(k_freq_hashed.keys()) - to_keep
    for d in discard:
        k_freq_hashed.pop(d)
    return k_freq_hashed


#%% [markdown]
# ### Overview of the itemsets
itemset, itemset_hashed, encodings, encodings_reverse = import_data('data\\movies.txt')
# output_support(itemset_hashed)
itemset = encode_dataset(itemset, encodings)

#%%
support=500
updated_itemset = itemset
f_sets = []
for i in range(1,20):
    print(f'Iteration: {i}')
    encoded_hash = shingle_hash(updated_itemset, i, support)
    # graph = generate_graph(encoded_hash)
    # ipd.Image(graph.create_png())
    
    if len(list(encoded_hash.keys())) <= 1:
        break
    updated_itemset = prune_itemset(encoded_hash, updated_itemset)
    f_sets.append(encoded_hash)

#%% 

f_sets[4]
#%%
fn = 'movies.txt'
with open(fn, 'w') as f:
    for s in f_sets:
        for k,v in s.items():
            f.write(f"{v}:")
            if len(k) < 2:
                f.write(f"{encodings_reverse[list(k)[0]]} ")
            else:
                for movie_id in k:
                    f.write(f"{encodings_reverse[movie_id]} ; ")
            f.write(f'\n')

#%%

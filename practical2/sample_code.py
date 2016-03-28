# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import cPickle

import util

TRAIN_DIR = "train"
TEST_DIR = "test"

call_set = set([])

tags_ignore = ['processes',
 'process',
 'thread',
 'all_section',]

def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        if call not in tags_ignore: 
            call_set.add(call)

# get universe of tags, ignoring useless tags        
def find_all_tags(start_index, end_index, direc="train"):
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break
            
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
    print "Ending file:", i
    
    # save call_set
    # np.save('valid_tags', call_set ) # ? array of set
    with open("valid_tags.pckl","w") as f:
        cPickle.dump( call_set, f )    
    
def create_data_matrix(start_index, end_index, direc="train"):
    X = None
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
        this_row = call_feats(tree)
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids

def call_feats(tree):
    # good_calls = ['sleep', 'dump_line']
    with open("valid_tags.pckl") as f:
        good_calls = cPickle.load(f)

    call_counter = {}
    for el in tree.iter():
        call = el.tag
        if call not in call_counter:
            call_counter[call] = 0
        else:
            call_counter[call] += 1

    call_feat_array = np.zeros(len(good_calls))
    for i, call in enumerate( good_calls ):
        call_feat_array[i] = 0
        if call in call_counter:
            call_feat_array[i] = call_counter[call]

    return call_feat_array

## Feature extraction
def main():
    X_train, t_train, train_ids = create_data_matrix(0, 5, TRAIN_DIR)
    X_valid, t_valid, valid_ids = create_data_matrix(10, 15, TEST_DIR)

    print 'Data matrix (training set):'
    print X_train
    print 'Classes (training set):'
    print t_train

    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).
    
def save_feature_data():
    X_train, t_train, train_ids = create_data_matrix(0, 4000, TRAIN_DIR)
    
    with open("train_features.pckl","w") as f:
        cPickle.dump( (X_train, t_train, train_ids), f )    
    
    X_test, t_test, test_ids = create_data_matrix(0, 4000, TEST_DIR)
    
    with open("test_features.pckl","w") as f:
        cPickle.dump( (X_test, t_test, test_ids), f )   
    
#if __name__ == "__main__":
#    main()

if False:
    X_train, t_train, train_ids = create_data_matrix(0, 5, TRAIN_DIR)
    X_valid, t_valid, valid_ids = create_data_matrix(10, 15, TRAIN_DIR)

    print 'Data matrix (training set):'
    print X_train
    print 'Classes (training set):'
    print t_train

    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).

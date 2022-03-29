import numpy as np
from sklearn.model_selection import train_test_split
from ._utils import read_csv, read_tsv, norm


def file2list(path,use_attri):
    data = read_csv(path)
    pairs = []
    labels = [0]*(len(data)-1)
    length = len(data[0])
    mid = int(length/2)
    if length % 2 == 1 :
        labels = [ int(x[length-1]) for x in data[1:] ]
    attri = [ x[2:] for x in data[0][:mid] ]
    if use_attri:
        attri = [ attri.index(x) for x in use_attri ]
    else:
        attri = [i for i in range(mid)]

    mid = int(length/2)
    for x in data[1:]:
        str1 = ""
        str2 = ""
        for j in attri:
            str1 = str1 + x[j]
            str2 = str2 + x[mid+j]
        
        pair = str1 + " [SEP] "+ str2
        pairs.append(norm(pair))

    print("****** Data Example ****** ")
    print("Entity pairs: ",pairs[:10])
    print("Label: ", labels[:10])
    return pairs, labels


def load_data(path, use_attri=None, valid_rate=None):
    # read data from path: line[left.title,left.name, ...  Tab right.title,right.name, ...  Tab label]
    pairs, labels = file2list(path,use_attri)

    # split to train/valid
    if valid_rate:
        train_x, valid_x, train_y, valid_y = train_test_split(pairs, labels,
                                                                test_size=valid_rate,
                                                                stratify=labels,
                                                                random_state=0)
        return train_x, valid_x, train_y, valid_y                                                   
    else:
        return pairs, labels

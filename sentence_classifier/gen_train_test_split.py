'''A tool to split data into a training set and a testing set.'''
import os
import csv
import random

approximate_test_fraction = 0.2

# Setting up the unrated file reader.
frompath = os.path.abspath(os.path.join('.', 'data', 'rated_data.csv'))
fromfile = open(frompath, 'r', newline='', encoding='utf-8')
fromcsv = csv.reader(fromfile)

# Setting up the rated file writer.
trainpath = os.path.abspath(os.path.join('.', 'data', 'train_data.csv'))
trainfile = open(trainpath, 'w', newline='', encoding='utf-8')
traincsv = csv.writer(trainfile)

# Setting up the skipped file writer.
testpath = os.path.abspath(os.path.join('.', 'data', 'test_data.csv'))
testfile = open(testpath, 'w', newline='', encoding='utf-8')
testcsv = csv.writer(testfile)

for row in fromcsv:
    if random.random() < approximate_test_fraction:
        testcsv.writerow(row)
    else:
        traincsv.writerow(row)

fromfile.close()
trainfile.close()
testfile.close()

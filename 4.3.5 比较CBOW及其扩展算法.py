
'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/27 15:02
@Author  : Zhangyunjia
@FileName: 4.3.5 比较CBOW及其扩展算法.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''
import os
import csv
from matplotlib import pylab
import numpy as np

cbow_loss_path = os.path.join('cbow_losses.csv')
with open(cbow_loss_path, 'rt') as f:
    reader = csv.reader(f,delimiter=',')
    for r_i,row in enumerate(reader):
        if r_i == 0:
            cbow_loss =  [float(s) for s in row]

skip_gram_loss_path=os.path.join('skip_losses.csv')
with open(skip_gram_loss_path, 'rt') as f:
    reader = csv.reader(f,delimiter=',')
    for r_i,row in enumerate(reader):
        if r_i == 0:
            skip_gram_loss =  [float(s) for s in row]

cbow_loss_unigram_path=os.path.join('cbow_loss_unigram.csv')
with open(cbow_loss_unigram_path, 'rt') as f:
    reader = csv.reader(f,delimiter=',')
    for r_i,row in enumerate(reader):
        if r_i == 0:
            cbow_loss_unigram =  [float(s) for s in row]

cbow_loss_unigram_subsampled_path=os.path.join('cbow_loss_unigram_subsampled.csv')
with open(cbow_loss_unigram_subsampled_path, 'rt') as f:
    reader = csv.reader(f,delimiter=',')
    for r_i,row in enumerate(reader):
        if r_i == 0:
            cbow_loss_unigram_subsampled =  [float(s) for s in row]

pylab.figure(figsize=(15,5))  # in inches

# Define the x axis
x = np.arange(len(skip_gram_loss))*2000

# Plotting standard CBOW loss, CBOW loss with unigram sampling and
# CBOW loss with unigram sampling + subsampling here in one plot
pylab.plot(x, cbow_loss, label="CBOW",linestyle='--',linewidth=2)
pylab.plot(x, cbow_loss_unigram, label="CBOW (Unigram)",linestyle='-.',linewidth=2,marker='^',markersize=5)
pylab.plot(x, cbow_loss_unigram_subsampled, label="CBOW (Unigram+Subsampling)",linewidth=2)

# Some text around the plots
pylab.title('Original CBOW vs Various Improvements Loss Decrease Over-Time',fontsize=24)
pylab.xlabel('Iterations',fontsize=22)
pylab.ylabel('Loss',fontsize=22)
pylab.legend(loc=1,fontsize=22)

# Use for saving the figure if needed
pylab.savefig('loss_cbow_vs_all_improvements.png')
pylab.show()

'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/26 17:37
@Author  : Zhangyunjia
@FileName: 4.2 比较skip-gram算法和CBOW算法.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''
import os
import csv
import numpy as np
from matplotlib import pylab

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

pylab.figure(figsize=(15,5))  # in inches

# Define the x axis
x = np.arange(len(skip_gram_loss))*2000

# Plot the skip_gram_loss (loaded from chapter 3)
pylab.plot(x, skip_gram_loss, label="Skip-Gram",linestyle='--',linewidth=2)
# Plot the cbow_loss (loaded from chapter 3)
pylab.plot(x, cbow_loss, label="CBOW",linewidth=2)

# Set some text around the plot
pylab.title('Skip-Gram vs CBOW Loss Decrease Over Time',fontsize=24)
pylab.xlabel('Iterations',fontsize=22)
pylab.ylabel('Loss',fontsize=22)
pylab.legend(loc=1,fontsize=22)

# use for saving the figure if needed
pylab.savefig('loss_skipgram_vs_cbow.png')
pylab.show()
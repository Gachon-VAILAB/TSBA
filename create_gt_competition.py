#-*- coding: utf-8 -*-
"""make imgpath - label list for competition train dataset"""

f = open('train.csv')
r = open('data/gt_competition.txt','w')
for i,line in enumerate(f):
    if(i==0):
        continue
    data = line.split(',')
    r.write(data[0][2:7]+'_competition'+data[0][7:]+'\t'+data[1])



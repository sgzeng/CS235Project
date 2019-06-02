#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import pandas as pd

def mergeCSV(in1, in2, outputPath):
    df1 = pd.read_csv(in1, header=0)
    df2 = pd.read_csv(in2, header=None)
    colNum = df2.shape[1]
    columns = []
    for i in range(colNum):
        columns.append('pc'+str(i))
    df2.columns = columns
    output = df1.join(df2, how='right')
    print(outputPath)
    output.to_csv(outputPath)

'''
This script merges two csv files recursively into one into output folder.
For example, if we have two folders in root dir:
./input1/
./input2/
each input* folder contains csv files, This script merges two csv files 
having same filename recursively into output folder. To run:
# pip install wheel
# pip install pandas
# python mergeCSV.py ./input1 ./input2

'''
if __name__=="__main__":
    fileMap = {}
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    outputDir = "output"
    if not (os.path.isdir(input1) and os.path.isdir(input2)):
        print("arg[1] and arg[2] need to be the input folders")
        exit(0)
    if os.path.isdir(outputDir):
        os.system('rm -rf ' + outputDir)
    os.system('mkdir -p '+ outputDir)
    for file in os.listdir(input1):
        if file not in fileMap:
            fileMap[file] = {0:"",1:""}
        if not file.endswith('.csv'):
            continue
        fileMap[file][0]=(os.path.join(input1,file))
    for file in os.listdir(input2):
        if file not in fileMap:
            fileMap[file] = {0:"",1:""}
        if not file.endswith('.csv'):
            continue
        fileMap[file][1]=(os.path.join(input2,file))
    for f in fileMap:
        if fileMap[f][0] and fileMap[f][1]:
            mergeCSV(fileMap[f][0], fileMap[f][1], os.path.join(outputDir,f))

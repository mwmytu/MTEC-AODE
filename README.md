# MTEC-AODE

# 1.Title

Multi-time Scale Augmented Neural ODEs Graph Neural for Traffic Flow Prediction with Elastic Channel Variation

# 2.Dataset

PEMSD3: http://pems.dot.ca.gov/

PEMSD4: http://pems.dot.ca.gov/

PEMSD7: http://pems.dot.ca.gov/

PEMSD7L: http://pems.dot.ca.gov/

PEMSD7M: http://pems.dot.ca.gov/

PEMSD8: http://pems.dot.ca.gov/

METR-LA: https://paperswithcode.com/dataset/metr-la

PeMS-BAY: https://paperswithcode.com/dataset/pems-bay

PEMSD(4,8)(3,5,7): change from utils ---> from random_access

# 3.Baselines

Some of these model comparison experiments were conducted on the LibCity(https://bigscity-libcity-docs.readthedocs.io/en/latest/index.html) framework.


# 4.Code

run the  __run_aode.py__ file.

# 5.Overall framework

![PNG](/Overall_Architecture.png)

# 6.Requirement

* python 3.7
* torch 1.7.0+cu101
* torchdiffeq 0.2.2
* fastdtw 0.3.4
* pip install fastdtw==0.3.4




# DGCL:Distance-wise and Graph Contrastive Learning for  Medication Recommendation


## Overview
This repository contains code necessary to run DGCL. DGCL is an end-to-end model mainly based on graph convolutional networks (GCN). We first devise a
correction loss to penalize the distance between the probability distributions of current cases and past medical history to obtain a link
between patients and past medical history. Then we propose a new medication comparison framework that can encode external
knowledge of DDI and guide the model to differentiate interactions in medication combinations.


## Requirements
- Pytorch >=0.8
- Python >=3.5

##How to run the code
## First, data preprocessing
Our process of preparing data just follows SafeDrug with slight modifications. Put the files of MIMIC III and II into the 'data' dir as below:
In ./data, you can find the well-preprocessed data in pickle form. Also, it's easy to re-generate the data as follows:
1.  download [MIMIC data](https://mimic.physionet.org/gettingstarted/dbsetup/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
2.  download [DDI data](：https://pan.baidu.com/s/1Ey-XBZfxwrvIuD7mvNz2VQ  Extraction code ：1234) and put it in ./data/
3.  run code **./data/preprocess.ipynb**

 
 
 ###
 ```
 python train_DGCL.py --model_name DGCL --ddi# training with DDI knowledge

 ```
 
## Cite 

Please cite our paper if you use this code in your own work

```
Acknowledgements
We thank all the people that provide their code to help us complete this project.

## A Novel Deep Learning Method for Identifying Antigen-Antibody Interactions (DeepAAI)

DeepAAI is an advanced deep learning-based tool for identifying antigen-antibody interactions, which was collected in Nature Machine Intelligence:

**Cite this article**

Zhang, J., Du, Y., Zhou, P. et al. Predicting unseen antibodies’ neutralizability via adaptive graph neural networks. Nature Machine Intelligence 4, 964–976 (2022). 

### Architecture
<div align="center">
<img src=/doc/img/model_architecture.png width=90% />
</div>


### Installation
```bash
pip install -r requirements.txt
```

### Training
Before training, all the files (```*.7z```) in the ```dataset/corpus/processed_mat/``` need to be decompressed. These decompressed files are processed datasets which are the input of the models.

Execute the following scripts to train antigen-antibody neutralization model with kmer features on the HIV dataset.
```bash
python model_trainer/deep_aai_kmer_embedding_cls_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody IC50 prediction model with kmer features on the HIV dataset.
```bash
python model_trainer/deep_aai_kmer_embedding_reg_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody neutralization model with kmer and pssm features on the HIV dataset.
```bash
python model_trainer/deep_aai_kmer_pssm_embedding_cls_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody IC50 prediction model with kmer and pssm features on the HIV dataset.
```bash
python model_trainer/deep_aai_kmer_pssm_embedding_reg_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody neutralization model on the SARS-CoV-2 dataset.
```bash
python model_trainer/deep_aai_kmer_embedding_cov_cls_trainer.py --mode train
```

Hyper-parameter in DeepAAI: 
| Parameter | Value | 
| ----  | ----  |
| Dropout| 0.4 | 
| Adj L1 loss | 5e-4 | 
| Param L2 loss | 5e-4 |
| Amino embedding size | 7 |
| Hidden size | 512 |
| Learning rate | 5e-5 |


Execute the following scripts to train antigen-antibody neutralization model by AG-Fast-Parapred on the HIV dataset.
```bash
python model_trainer/baseline_ag_fast_parapred_cls_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody IC50 prediction model by AG-Fast-Parapred on the HIV dataset.
```bash
python model_trainer/baseline_ag_fast_parapred_reg_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody neutralization model by Parapred on the HIV dataset.
```bash
python model_trainer/baseline_parapred_cls_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody IC50 prediction model by Parapred on the HIV dataset.
```bash
python model_trainer/baseline_parapred_reg_trainer.py --mode train
```


### Preprocessing dataset
The data pre-processing module is in the folder of ```processing/```. There are three sub-folders in the processing folder, ```hiv_cls```, ```hiv_reg```, and ```cov_cls```. The pre-processing can be understood by following the scripts of processing.py as well as the *.py under processing/hiv_cls/, processing/hiv_reg/, and processing/cov_cls/.

- ```processing/hiv_cls/``` includes the scripts and the source data that generate the dataset for HIV classification. The source file is ```dataset_hiv_cls.xlsx```, which contains four fields ```antibody_seq```, ```virus_seq```, ```label```, and ```split```. ```processing/hiv_cls/corpus/cls``` contains data indices. The generated dataset for HIV classification will be under ```processing/hiv_cls/corpus/processed_mat```. 

- ```processing/hiv_reg/``` includes the scripts and the source data that generate the dataset for HIV regression. The source file is ```dataset_hiv_reg.xlsx```, which contains four fields ```antibody_seq```, ```virus_seq```, ```label```, and ```split```. ```processing/hiv_cls/corpus/cov_cls``` contains data indices. The generated dataset for HIV regression will be under ```processing/hiv_reg/corpus/processed_mat```. 

- ```processing/cov_cls/``` includes the scripts and the source data that generate the dataset for SARS-CoV2 classification. The source file is ```dataset_cov_cls.xlsx```, which contains four fields ```antibody_seq```, ```virus_seq```, ```label```, and ```split```. ```processing/cov_cls/corpus/cov_cls``` contains data indices. The generated dataset for SARS-CoV2 will be under ```processing/cov_cls/corpus/processed_mat```. 

Noted that for each SARS-CoV-2 variant (SARS-CoV2_WT, SARS-CoV2_Alpha, SARS-CoV2_Beta, SARS-CoV2_Gamma, SARS-CoV2_Delta), five sequences were sampled, respectively. The unseen test includes SARS-CoV2_Omicron.

Under ```processing/hiv_cls/```, ```processing/hiv_reg/```, and ```processing/cov_cls/```, there are ```processing.py```s. Lines 62-82 in each ```processing.py``` correspond to how to convert the sequence into kmer, one-hot, pssm, etc.

- Lines 62-66: one-hot
- Lines 68-69: pssm
- Lines 71-75: amino_num
- Lines 77-82: k-mer-whole

The pssm needs to be obtained from the POSSUM ([Wang, J. et al. Possum: a bioinformatics toolkit for generating numerical sequence feature descriptors based on pssm profiles. Bioinformatics 33, 2756–2758 2017](https://academic.oup.com/bioinformatics/article/33/17/2756/3813283)) and placed in the pssm folder. We select the Uniref50 database to generate PSSMs. 

Execute the following scripts to process the HIV dataset for classification.
```bash
python processing/hiv_cls/processing.py
```

Execute the following scripts to process the HIV dataset for regression.
```bash
python processing/hiv_reg/processing.py
```

Execute the following scripts to process the SARS-CoV-2 dataset.
```bash
python processing/cov_cls/processing.py
```

There is no additional criteria for filtering the data of the other datasets. For more details of data collection and features, please see the in the subsections of **Data** and **Feature** of the **Method** section in Page 9 ~ 10 of the manuscript.





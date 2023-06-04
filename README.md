## A Novel Deep Learning Method for Identifying Antigen-Antibody Interactions (DeepAAI)

DeepAAI is an advanced deep learning-based tool for identifying antigen-antibody interactions.

For making DeepAAI available at no cost to the community, we have set a **[web service](https://aai-test.github.io/)** predicting antigen-antibody interactions. 
[https://aai-test.github.io/]( https://aai-test.github.io/)

### Architecture   
![](/doc/img/model_architecture.png)

### Installation
```bash
pip install -r requirements.txt
```

### Training
Before training, all the files (```*.7z```) in the ```dataset/corpus/processed_mat``` need to be decompressed.

Execute the following scripts to train antigen-antibody neutralization model on the HIV dataset.
```bash
python model_trainer/deep_aai_kmer_embedding_cls_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody neutralization model on the SARS-CoV-2 dataset.
```bash
python model_trainer/deep_aai_kmer_embedding_cov_cls_trainer.py --mode train
```

### Processing dataset
Execute the following scripts to process the HIV dataset.
```bash
python processing/hiv_cls/processing.py
```

Execute the following scripts to process the SARS-CoV-2 dataset.
```bash
python processing/cov_cls/processing.py
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





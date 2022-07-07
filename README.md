## A Novel Deep Learning Method for Identifying Antigen-Antibody Interactions (DeepAAI)
DeepAAI is an advanced deep learning-based tool for identifying antigen-antibody interactions. We devise an automatically learned virtual graph to address antibodies’ high variability. The virtual graph connects seen and unseen antibodies by quantitating functional similarity based on the supervised signals from two downstream tasks: binary neutralization prediction and IC50 estimation.

We provided clear instructions on installing and running the program with the dataset specified the software and hardware requirements and exposed the modifiable settings as input parameters. 

For making DeepAAI available at no cost to the community, we have set a **[web service](https://aai-test.github.io/)** predicting antigen-antibody interactions. 
[https://aai-test.github.io/]( https://aai-test.github.io/)




### Architecture   
![](/docs/images/2.pdf)

### Installation
```bash
pip install -r requirements.txt
```


### Usage  
Execute the following scripts to predict the probability 
of antigen-antibody binding
```bash
# load parameters for evaluation
python deep_aai_kmer_embedding_cls_evaluate.py --infile test_data.csv --outfile pred_result.csv
```
### Key options of this scrips:  
- `infile`: CSV file to be predicted (default=test_data.csv).   

<table>
  <tr>
    <td>virus_seq</td>
    <td>heavy_seq</td>
    <td>light_seq</td>
    <td>label_10</td>
  </tr>
  <tr>
    <td>MRVTGIRRNCRH...</td>
    <td>QKQLVESGGGVV...</td>
    <td>QSVLTQPPSVSA...</td>
    <td>1 or 0</td>
  </tr>
</table>
&nbsp&nbsp&nbsp&nbsp If input file not contain 'label_10' column, the evaluation of prediction results will be skipped.  
<br/>
&nbsp

- `outfile`: Prediction results (default=pred_result.csv)
      

### Descriptions  
The most important files in this projects are as follow:
- dataset: 
  - `abs_dataset_cls.py`: The object for load the HIV classification data set. 
  - `k_mer_utils.py`: Create K-mer feature. 
- baseline_trainer: Train scrips for DeepAAI and some baselines. 
- models: Implementation of DeepAAI and all baselines.
- save_model_param_pred: Saved model parameters.


### Training
Execute the following scripts to train antigen-antibody binding model.
```bash
python model_trainer/deep_aai_kmer_embedding_cls_trainer.py --mode train
```
Hyper-parameter in DeepAAI: 
| Parameter | Value | 
| ----  | ----  |
| Dropout| 0.4 | 
| Adj L1 loss | 5e-4 | 
| Param L2 loss | 5e-4 |
| CNN kernel | 7, 9, 11 |
| CNN channel | 256 |
| Amino embedding size | 7 |
| Hidden size | 512 |
| Learning rate | 5e-5 |





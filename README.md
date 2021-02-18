## DeepAAI
code for A Novel Deep Learning Framework for Predicting Antigen-Antibody Interaction


Architecture   
![](/docs/images/model_architecture.png)

Installation
```bash
pip install -r requirements.txt
```


Usage  
Execute the following scripts to predict the probability 
of antigen-antibody binding
```bash
# load parameters for evaluation
python deep_aai_kmer_embedding_cls_evaluate.py --infile test_data.csv --outfile pred_result.csv
```
Key options of this scrips:  
- `infile`: CSV file to be predicted (default=test_data.csv).   

<style> table th:first-of-type { width: 100px; } </style>
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

- `outfile`: Prediction results (default=pred_result.csv)
      

Descriptions  
The most important files in this projects are as follow:
- dataset: 
  - `abs_dataset_cls.py`: The object for load the HIV classification data set. 
  - `k_mer_utils.py`: Create K-mer feature. 
- baseline_trainer: Train scrips for DeepAAI and some baselines. 
- models: Implementation of DeepAAI and all baselines.
- save_model_param_pred: Saved model parameters




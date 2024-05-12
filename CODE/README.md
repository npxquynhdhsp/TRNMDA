# TRNMDA
A deep metric learning based method for predicting miRNA-disease associations

## Requirements
  * python==3.8.18
  * numpy==1.24.3
  * scikit-learn==1.3.0
  * pandas==2.0.3
  * tensorflow==2.4.0
  * xgboost==2.0.2
  * xlrd==1.2.0
  * openpyxl==3.1.2

## File
### data
  All data input files needed to run the model are in the folder ```../IN```, which contain folders:
  * ```ORI DATA```: contains original input data which are used for HMDDv2.0 and HMDDv3.2 in the same name folders, respectively.
    - miRNA functional similarity matrix.txt: MiRNA functional similarity
    - disease semantic similarity matrix 1.txt and disease semantic similarity matrix 2.txt: Two kinds of disease semantic similarity
    - known disease-miRNA association number.txt: Validated mirNA-disease associations
    - disease number.txt: Disease id and name
    - miRNA number.txt: MiRNA id and name
  * ```Q18``` and ```Q18_HMDD3``` contain proceeded data files for HMDDv2.0 and HMDDv3.2, respectively.
    - Folder ```Q18/kfold/``` contains data input files respective for folds in 5-fold-CV on HMDDv2.0.
    - Folder ```Q18_HMDD3/kfold/``` contains data input files respective for folds in 5-fold-CV on HMDDv3.2.
    - Folder ```Q18_HMDD3/dis_k/``` contains data input files respective for case studies on HMDDv3.2.
      Each folder includes proceeded files:
      + SR_FS*: Integrated miRNA similarity matrix
      + SD_SS*: Integrated disease similarity matrix
      + y_train*: Training human MDAs matrix
      + y_4loai*: Human MDAs matrix which are ussed for train and test phase.
### result
  The predictive scores in 5-fold-CV and case studies are files ```*prob_trbX*``` in the folder ```Q23_TripletNetwork/OUT Q_*/*/Q18/Results/Combination/```.
### code
  * Folder ```GENERAL_UTILS``` contains files for data processing.
  * For 5-fold-CV in HMDD2 or HMDD3 and case studies in HMDD3:
    - params.py: For changing parameters
    - model.py: Structure of the model
    - train_TripletNet1_*.py, train_TripletNet2_*.py: Train triplet networks
    - kethop_kfold_va_dis_k_2_loai.py: Train the final model.
## Usage
  * Download code and data.
  * Because of the big size of dataset, data in github is uploaded for using one repeat time running. You can edit code to run in one repeat time. For further data, please feel free send email to npxquynh@hueuni.edu.vn.
  * How to run:
     - For 5-fold-CV in HMDD2 or HMDD3:
      1. Choose dataset and type of evaluation. Default: ```HMDD3``` and ```kfold```. If you want to change parameters, edit in the file ```params.py```. 
        1.1. Choose dataset:
          ```db = 'HMDD2'``` or ```db = 'HMDD3'```
        1.2. Choose type of evaluation:
          ``` type_eval = 'kfold' ```
      2. Run for 5-fold-CV:
        2.1. Run ```train_TripletNet1_kfold.py```
        2.2. Run ```train_TripletNet2_kfold.py```
        2.3. Run ```kethop_kfold_va_dis_k_2_loai.py```.
    - For case studies in HMDD3:
      1. Choose dataset and type of evaluation.
        1.1. Choose dataset:
          ```db = 'HMDD3'```
        1.2. Choose type of evaluation:
          ``` type_eval = 'dis_k' ```
      2. Run for case studies:
        2.1. Run ```train_TripletNet1_dis_k.py```
        2.2. Run ```train_TripletNet2_dis_k.py```
        2.3. Run ```kethop_kfold_va_dis_k_2_loai.py```.

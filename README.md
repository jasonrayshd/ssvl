### Notice:  
(1) Pretraining file entrypoint:  run_mae_pretraining.py  engine_for_pretraining.py
(2) Fintuning file entrypoint:  run_class_finetuning.py  engine_for_finetuning.py
(3) Bash scripts are written for runninng different tasks:  
    e.g. local_pretrain_ts_epic.sh is used to pretrain our method on local machine. it will read config from config/pretrain_ts_epic55.yml

(4) Model definition for pretraining: modeling_pretrain.py  
    Model definition for fintuning: modeling_finetune.py  

(5) Dataset file: ego4d.py, epickitchens.py  

#### (6) Logics of online extracting flow images from ego4d/epic-kitchens  
    a. For each subprocess (identified by global rank), initialize a multiprocessing.Manager object and register flow extracting function  
    b. start the Manager object as a seperate process on each gpu and pass it to Dataset object as a parameter  
    c. extract optical flow in function Dataset.__getitem__  


### [2022.09.07]
To run pretraining experiments under our latest setting (two stream setting):  
(1) modify necessary path parameters (e.g., path of epic55 dataset and output directory) in configuration file ./config/pretrain_ts_epic55.yml  
(2) modify absolute path of the configuration file ./config/pretrain_ts_epic55.yml in script ./scripts/local_pretrain_ts_epic.sh  
(3) run following command in ./scripts:  
```
# single node with 4 gpus by default
CUDA_VISIBLE_DEVICES="0,1,2,3" bash local_pretrain_ts_epic.sh 0 0.0.0.0  
```

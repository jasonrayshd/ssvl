## File organization:  
(1) **Pretraining file entrypoint**:  run_mae_pretraining.py  engine_for_pretraining.py  
(2) **Fintuning file entrypoint**:  run_class_finetuning.py  engine_for_finetuning.py  
(3) **Model definition for pretraining**: modeling_pretrain.py  
    **Model definition for fintuning**: modeling_finetune.py  
(4) **Dataset file**: ego4d.py, epickitchens.py  


## Multimodal Pretraining
### Pretraining on Egoclip

**Step 1**: Modify bash script as needed - File: scripts/temp_pretrain_multimodal.sh

(1) modify *gpu per node* setting: set "nproc_per_node" to required gpu number  

(2) modify *name* of the experiment: set "--name" to a different name
the name of the experiment determines the name of the output directory that stores model checkpoints. Therefore, to avoid overwrting files produced by other experiments, be sure to use a different name for each experiment.

(3) (optional) modify *project* of the experiment: set "--project" to related project.
the parameter "project" is the name of the wandb project. wandb is a online logging tools that upload required hyper-parameters/results online. Logs are grouped via different projects specified by user. You could specify a different project name as needed.

for example:
```
pretrain_multimodal_epic55: experiments that pretrains on epic55/egoclip using multimodal pretraining scheme
```
**you could disable the wandb logging by specifying "--debug" and debug code locally without recording trivial experiments online.**


**Step 2**: Modify configuration file as needed - File: config/cluster/pretrain_multimodal_egoclip.yml

e.g. Specify a different model architecture by assigning new value to "model"


**Final step**: run bash script  

e.g. run multimodal pretraining locally on egoclip with 4 GPUs  

```bash
cd scripts
CUDA_VISIBLE_DEVICES="0,1,2,3" bash temp_pretrain_multimodal.sh 0 0.0.0.0 ../config/cluster/pretrain_multimodal_egoclip.yml
# parameter of bash script
# $1: node rank of current process
# $2: ip address of main process whose node rank is 0
# $3: path to configuration file
# be aware that when running bash script, current working directory is ./scripts. 
```

### Pretraining on Epic55
**Step 1:** same as pretraining on egoclip 

**Step 2**: Modify configuration file as needed - File: config/cluster/pretrain_multimodal_epic55.yml

e.g. Specify a different model architecture by assigning new value to "model"

**Final step**: run bash script

e.g. run multimodal pretraining locally  on epic55 with 4 GPUs  

```bash
cd scripts
CUDA_VISIBLE_DEVICES="0,1,2,3" bash temp_pretrain_multimodal.sh 0 0.0.0.0 ../config/cluster/pretrain_multimodal_epic55.yml
# parameter of bash script
# $1: node rank of current process
# $2: ip address of main process whose node rank is 0
# $3: path to configuration file
# be aware that when running bash script, current working directory is ./scripts. 
```

### run experiment on cluster

e.g. run multimodal pretraining scheme on egoclip with 2x8V100GPUs on clusters

```
# cluster configuration file
description: ...

env_defaults:
  NODES: 2
  GPUS: 8
  MEM: 24

target:
  service: aml
  name: *CLUSTER_NAME*

environment:
  image: jiachenlei/ssvl-vmae_amlt8.1.1:latest

code:
  local_dir: */PATH/TO/ssvl/videomae*
  storage_id: default

storage:
  data:
    storage_account_name: compassresearch
    container_name: shuang
    mount_dir: /mnt/shuang

  output:
    storage_account_name: compassresearch
    container_name: shuang
    mount_dir: /mnt/shuang

jobs:
  # pretrain on egoclip with multimodal scheme
  - name: multimodal_preegoclip_A0
    sku: ${NODES}x${MEM}G${GPUS}
    command:
      - >-
          python run_mae_pretraining.py
          --overwrite command-line 
          --dist_on_itp
          --config ./config/cluster/pretrain_multimodal_egoclip.yml
          --project pretrain_multimodal_epic55
          --name multimodal_preegoclip_A0
    aml_mpirun:
      process_count_per_node: ${GPUS}
      communicator: "OpenMpi"
    submit_args:
      container_args:
        shm_size: 50g

  # pretrain on epic55 with multimodal scheme
  - name: multimodal_preegoclip_A0
    sku: ${NODES}x${MEM}G${GPUS}
    command:
      - >-
          python run_mae_pretraining.py
          --overwrite command-line 
          --config ./config/cluster/pretrain_multimodal_epic55.yml
          --project pretrain_multimodal_epic55
          --name multimodal_preepic55_A8
    aml_mpirun:
      process_count_per_node: ${GPUS}
      communicator: "OpenMpi"
    submit_args:
      container_args:
        shm_size: 50g
```

## Fintuning
There are 5 downstream tasks in total that are supported:  
(1) **OSCC**: binary classification task that requries model predicting whether object state change occurs in a given clip.  
(2) **PNR**: temporal localization task that requires model predicting the index of the frame where the object state change occurred in a given clip.  
(3) **Long-Term Anticipation**: given an observed clip, predict a sequantial of (e.g. 20) future activities that each activity is composed of action(verb) and the name of an object(noun) to be interacted with.  
(4) **Short-Term Anticipation**: given an observed clip, predict all possbile future actions that might take place in a short time, possibilities of the action take place, time that the action take place and position of the object to be interacted with  
(5) **Future Hands prediction**: given an observed clip, predict the positions of left and right hands in predetermined 5 future moments  

### Finetuning on OSCC


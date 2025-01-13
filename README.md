# Full-atom d-peptide co-design with flow matching

Our codebase is developed on top of [FrameFlow](https://github.com/microsoft/frame-flow), [MultiFlow](https://github.com/jasonkyuyim/multiflow), 
[PepFlow](https://github.com/Ced3-han/PepFlowww), and [ByProt](https://github.com/BytedProtein/ByProt).

<p align="center"><img src="dflow.png" alt="drawing" width="900"/></p>

## Installation

### Step 1: Set Up the Environment
```bash
# Create and activate a conda environment with dependencies
conda env create -f env.yml
conda activate dflow

# Install the local package
pip install -e .

# Additional dependencies
pip install easydict lmdb
```
### Step 2: Install torch-scatter
Manually install `torch-scatter` based on your PyTorch version. For PyTorch `2.4.1` and CUDA `12.1` (H100 GPUs), run:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```
For other versions, find your PyTorch version with:
```bash
# Find your installed version of torch
python
>>> import torch
>>> torch.__version__
# Example: torch 2.4.1+cu121
```

> [!WARNING]  
> If you encounter the error below from DeepSpeed:
> ```bash
> ModuleNotFoundError: No module named 'torch._six'
> ```
> Replace `from torch._six import inf` with `from torch import inf`in these files: 
> * `/path/to/envs/site-packages/deepspeed/runtime/utils.py`
> * `/path/to/envs/site-packages/deepspeed/runtime/zero/stage_1_and_2.py`
> where `/path/to/envs` is your path. 

## Data
### Pretraining Datasets

Pretrain datasets are host on Zenodo [here](https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTk2YjlmLTM4OTUtNGVhYi1hODcxLWE1ZjExOTczY2IzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDY5ZDUzYjVjMTNhNDllMDYxNmI3Yjc2NjcwYjYxZiJ9.C2eZZmRu-nu7H330G-DkV5kttfjYB3ANozdOMNm19uPahvtLrDRvd_4Eqlyb7lp24m06e4OHhHQ4zlj68S1O_A).
Download the following files,
* `real_train_set.tar.gz` (2.5 GB)
* `synthetic_train_set.tar.gz` (220 MB)
* `test_set.tar.gz` (347 MB)

Next, untar the files
```bash
# Uncompress training data
mkdir train_set
tar -xzvf real_train_set.tar.gz -C train_set/
tar -xzvf synthetic_train_set.tar.gz -C train_set/

# Uncompress test data
mkdir test_set
tar -xzvf test_set.tar.gz -C test_set/
```
The resulting directory structure should look like
```bash
<current_dir>
├── train_set
│   ├── processed_pdb
|   |   ├── <subdir>
|   |   |   └── <protein_id>.pkl
│   ├── processed_synthetic
|   |   └── <protein_id>.pkl
├── test_set
|   └── processed
|   |   ├── <subdir>
|   |   |   └── <protein_id>.pkl
...
```
Our experiments read the data by using relative paths. Keep the directory structure like this to avoid bugs.

### PepMerge Dataset
PepMerge dataset is available on Google Drive [here](https://drive.google.com/drive/folders/1bHaKDF3uCDPtfsihjZs0zmjwF6UU1uVl?usp=sharing).
Downloading the following files:
+ `PepMerge_release.zip` (1.2GB)

The ```PepMerge_release.zip``` contains filtered data of peptide-receptor pairs, which is collected from 
[PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/db/1cta_A/) and [QBioLip](https://yanglab.qd.sdu.edu.cn/Q-BioLiP/Download). 
For example, in the folder ```1a0n_A```, the ```P``` chain in the PDB file ```1a0n``` is the peptide. 
In each sub-folder, FASTA and PDB files of the peptide and receptor are given. 
The postfix _merge means the peptide and receptor are in the same PDB file. 
The binding pocket of the receptor is also provided, where our model is trained to generate peptides based on the binding pocket.
When you run the code, it will automatically process the data and produce `pep_pocket_train_structure_cache.lmdb` and 
`pep_pocket_test_structure_cache.lmdb` in the default cache folder (i.e., `../pep_cache/`).



## Training
The command to run co-design training is the following, 
```bash
# pretrain 
python -W ignore dflow/experiments/train_se3_flows.py -cn pdb_codesign

# peptide training (1 GPU)
python -W ignore dflow/experiments/train_pep_flows.py

# DDP peptide training (e.g., 4 GPUs)
torchrun --nproc_per_node=4 dflow/experiments/train_pep_flows.py   

```
We use [Hydra](https://hydra.cc/) to maintain our configs. 
The training config is found here `multiflow/configs/pdb_codesign.yaml`.

Most important fields:
* `experiment.num_devices`: Number of GPUs to use for training. Default is 2.
* `data.sampler.max_batch_size`: Maximum batch size. We use dynamic batch sizes depending on `data.sampler.max_num_res_squared`. Both these parameters need to be tuned for your GPU memory. Our default settings are set for a 40GB Nvidia RTX card.
* `data.sampler.max_num_res_squared`: See above.


## Inference

Unpack the provided model weights:
```bash
unzip dlow.zip
```

The following generation task can be performed. 
```bash
# Unconditional Co-Design
python -W ignore dflow/experiments/inference_pep.py 
```
If you want to generate d-peptides, please set `x_mirror` to `True` in `dflow/configs/pep_codesign.yaml`. 

## Cite and Contact
If you find this paper and the corresponding code interesting and helpful, we would really appreciate it if you can cite the paper. Thank you! 😜
Moreover, we welcome any sort of relevant questions. If you have any questions, please contact [fangwu97@stanford.edu]. Thank you! :)
```markdown
@article{wu2024d,
  title={D-Flow: Multi-modality Flow Matching for D-peptide Design},
  author={Wu, Fang and Xu, Tinson and Jin, Shuting and Tang, Xiangru and Xu, Zerui and Zou, James and Hie, Brian},
  journal={arXiv preprint arXiv:2411.10618},
  year={2024}
}
```

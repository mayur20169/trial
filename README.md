## Get Started

### 1. Installation and Data Preparation
```
**A.** Create virtual-env.
```shell
conda create -n xmuda python=3.8
```

**B.** Install requirements
Use cuda-11.6
```
conda activate xmuda	
pip install -r requirements.txt
python setup.py develop
```

**C.** Download the dataset and create dataset infos
All the file will be organized as,
```
CMD
├── data
│   ├── xmu
│   │   │── ImageSets
|   |   |── label
|   |   |── seq**     
├── pcdet
├── tools
```

- Generate the data infos by running the following command: 
```python 
 python -m pcdet.datasets.xmu.xmu_dataset --func create_xmu_infos  --cfg_file tools/cfgs/dataset_configs/xmu/xmuda_dataset.yaml
```
- Generate gt_sampling_database by running the following command: 
```
python -m pcdet.datasets.xmu.xmu_dataset --func create_groundtruth_database  --cfg_file tools/cfgs/dataset_configs/xmu/xmu_dataset.yaml
```

# Experimental Results
## Training
- for singe gpu
```
python tools/train.py --cfg_file  tools/cfgs/xmu_ouster_models/centerpoint.yaml 
```

## Evaluation
- for singe gpu
```
python tools/test.py --cfg_file tools/cfgs/xmu_ouster_models/centerpoint.yaml --ckpt /path/to/your/checkpoint 
```

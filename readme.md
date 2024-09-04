# PSCA
Implementation of "Prototype-based Joint Category-Scale Adaptation for UAV Domain Adaptive Object Detection"
# Installation
Check [install.md](install.md) for installation instructions. The implementation of our anchor-free detector is heavily based on [FCOS](https://github.com/tianzhi0549/FCOS).

# Dataset
UAV-OD scenarios

1. [UAV-DT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5)
   - Download the [object detection dataset](https://drive.google.com/file/d/1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc/view) and its [detection annotations](https://drive.google.com/file/d/19498uJd7T9w4quwnQEy62nibt3uyT9pq/view) and [weather annotations](https://drive.google.com/file/d/1qjipvuk3XE3qU3udluQRRcYuiKzhMXB1/view)
   - According to the settings of [UAVDT-weather.md](dataset/utils/UAVDT_WEATHER.md), the dataset is divided into daylight, night, foggy

2. [VISDRONE](https://github.com/VisDrone/VisDrone-Dataset)

   - Download [training set](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view), [validation set](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view) and [test set](https://drive.google.com/file/d/1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V/view?usp=drive_open), a total of 8599 images
   - All images are integrated into an images folder, and the corresponding annotations are integrated into the annotations folder.
   - According to the [daylight_train.txt](dataset/visdrone/daylight/train.txt)、[daylight_val.txt](dataset/visdrone/daylight/val.txt) and [night_train.txt](dataset\visdrone\night\train.txt)、[night_val.txt](dataset\visdrone\night\val.txt) we provided, the dataset is divided into a daylight dataset and a night dataset.

   We provide [annotation and  images catalog ](dataset) to facilitate the organization of the dataset .

After the preparation, the dataset should be stored as follows:

```
[DATASET_PATH]
└─ UAV-DT
   └─ daylight
      └─ train
      └─ val
      └─ train.json
      └─ val.json
   └─ night
      └─ train
      └─ val
      └─ train.json
      └─ val.json
   └─ fog
      └─ train
      └─ val
      └─ train.json
      └─ val.json
└─ Visdrone
   └─ daylight
      └─ train
      └─ val
      └─ train.json
      └─ val.json
   └─ night
      └─ train
      └─ val
      └─ train.json
      └─ val.json
```



# Format and Path
Before training, please check paths_catalog.py and enter the correct data path for:

- `DATA_DIR`
- `visdrone_daylight_train_cocostyle`, `visdrone_night_train_cocostyle`  and `visdrone_night_val_cocostyle` (for daylight ->night on visdrone).
- `UAVDT_daylight_train_cocostyle`, `UAVDT_night_train_cocostyle`  and `UAVDT_night_val_cocostyle` (for daylight ->night on UAVDT).
- `UAVDT_daylight_train_cocostyle`, `UAVDT_fog_train_cocostyle`  and `UAVDT_fog_val_cocostyle` (for daylight ->foggy on UAVDT).

For example, if the datasets have been stored as the way we mentioned, the paths should be set as follows:

- Dataset directory (In L8):

  ```DATA_DIR = [DATASET_PATH]```

- Train and validation set directory for each dataset:

  ```
  ############# visdrone daylight 2 night ###################
  'visdrone_daylight_train_cocostyle':{
  	"img_dir": 'visdrone/daylight/train/images',
      "ann_file": 'visdrone/daylight/train.json',
      },
  'visdrone_night_train_cocostyle':{
      "img_dir": 'visdrone/night/train/images',
      "ann_file": 'visdrone/night/train.json',
      },
  'visdrone_night_val_cocostyle':{
      "img_dir": 'visdrone/night/val/images',
      "ann_file": 'visdrone/night/val.json',
      },
  ############# UAVDT daylight 2 night ###################
  'UAVDT_daylight_train_cocostyle':{
      "img_dir": 'UAV-DT/daylight/train/images',
      "ann_file": 'UAV-DT/daylight/train.json',
      },
  'UAVDT_night_train_cocostyle':{
      "img_dir": 'UAV-DT/night/train/images',
      "ann_file": 'UAV-DT/night/train.json',
      },
  'UAVDT_night_val_cocostyle':{
      "img_dir": 'UAV-DT/night/val/images',
      "ann_file": 'UAV-DT/night/val.json',
      },
  ############# UAVDT daylight 2 fog ###################
  'UAVDT_fog_train_cocostyle':{
      "img_dir": 'UAV-DT/fog/train/images',
      "ann_file": 'UAV-DT/fog/train.json',
      },
  'UAVDT_fog_val_cocostyle':{
      "img_dir": 'UAV-DT/fog/val/images',
      "ann_file": 'UAV-DT/fog/val.json',
      },
  ```

  



# Training
```
python tools/train_net_da.py --config-file configs_UAVDT/d2f_ga_MI_VGG_16_FPN_memory_4x_contrast_fusion.yaml
```


# Evaluation

```
python tools/test_net.py --config-file configs_UAVDT/d2f_ga_MI_VGG_16_FPN_memory_4x_contrast_fusion.yaml MODEL.WEIGHT xxx.pth
```




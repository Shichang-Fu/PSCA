## Instructions for Organizing and Converting Data

### 1. Consolidate All Images into a Single Folder

Modify the `organise_image_folders.py` file by updating the `old_dir` and `output_dir` paths:

```python
old_dir = 'UAV-DT\\UAV-benchmark-M'
output_dir = 'UAV-DT\\dataset\\images'
```

After running this script, all images will be consolidated into `dataset/images`.

### 2. Convert Annotations to Bounding Box + Category Format

Annotations should be converted to the following format:

```
{bbox_left},{bbox_top},{bbox_width},{bbox_height},{category_id}
```

Update the `gt2txt.py` file with the correct paths for `gt_folder` and `output_folder`:

```python
gt_folder = 'UAV-DT\\UAV-benchmark-MOTD_v1.0\\GT'
output_folder = 'UAV-DT\\dataset\\txt'
```

Once this script is run, all annotations will be consolidated into `dataset/txt`.

### 3. Partition Images by Weather Conditions

Based on weather annotations, partition images into `daylight`, `night`, and `fog`. The weather annotations also include `train` and `test` splits. Run `python weather.py` twice: first for the train annotations and then for the test annotations.

### 4. Move TXT Files According to Weather Conditions

Run `python movetxtwithimages.py` to move TXT files into the corresponding weather folders. The directory structure should look like this:

```
UAV-DT
  └─ daylight
     └─ train
        └─ images
        └─ txt
     └─ val
        └─ images
        └─ txt
  └─ night
     └─ train
        └─ images
        └─ txt
     └─ val
        └─ images
        └─ txt
  └─ fog
     └─ train
        └─ images
        └─ txt
     └─ val
        └─ images
        └─ txt
```

### 5. Convert TXT Annotations to JSON Format

Run `python txt2json.py` to convert TXT annotations to JSON format.

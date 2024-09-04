import os
import shutil

# 定义属性和对应的类别
attribute_labels = ["daylight", "night", "fog", "low-alt", "medium-alt", "high-alt", "front-view", "side-view", "bird-view", "long-term"]

# 数据集根目录
images_path = "F:\\fsc_dataset\\UAV-DT\\dataset\\images"
# 属性文件根目录
dataset_root = "F:\\fsc_dataset\\UAV-DT\\M_attr\\train"

# 遍历所有属性文件
for filename in os.listdir(dataset_root):
    if filename.endswith("_attr.txt"):
        file_path = os.path.join(dataset_root, filename)
        
        # 读取属性文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 获取对应的图像文件夹名（假设属性文件名为M0101_attr.txt）
        image_folder = filename.replace("_attr.txt", "")
        image_folder_path = os.path.join(images_path, image_folder)

        # 获取目标属性文件夹
        for line in lines:
            values = line.strip().split(',')
            # 获取 "alt" 属性的值
            daylight = int(values[0])  # 对应 "daylight", "night", "fog"
            night = int(values[1] )
            fog = int(values[2])
            if daylight == 1:
                daylight_label = attribute_labels[0]
                destination_dir = os.path.join(dataset_root, daylight_label)
                os.makedirs(destination_dir, exist_ok=True)
                # 获取以 "M0101" 开头的图像路径
                source_folder = images_path
                for image_filename in os.listdir(source_folder):
                    if image_filename.startswith(image_folder):
                        source_path = os.path.join(source_folder, image_filename)
                        shutil.copy(source_path, destination_dir)
                break
            
            if night == 1:
                night_label = attribute_labels[1]
                destination_dir = os.path.join(dataset_root, night_label)
                os.makedirs(destination_dir, exist_ok=True)
                # 获取以 "M0101" 开头的图像路径
                source_folder = images_path
                for image_filename in os.listdir(source_folder):
                    if image_filename.startswith(image_folder):
                        source_path = os.path.join(source_folder, image_filename)
                        shutil.copy(source_path, destination_dir)
                break
            
            if fog == 1:
                fog_label = attribute_labels[2]
                destination_dir = os.path.join(dataset_root, fog_label)
                os.makedirs(destination_dir, exist_ok=True)
                # 获取以 "M0101" 开头的图像路径
                source_folder = images_path
                for image_filename in os.listdir(source_folder):
                    if image_filename.startswith(image_folder):
                        source_path = os.path.join(source_folder, image_filename)
                        shutil.copy(source_path, destination_dir)
                break

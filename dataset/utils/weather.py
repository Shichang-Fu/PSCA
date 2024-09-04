import os
import shutil

attribute_labels = ["daylight", "night", "fog", "low-alt", "medium-alt", "high-alt", "front-view", "side-view", "bird-view", "long-term"]

images_path = "UAV-DT\\dataset\\images"
dataset_root = "UAV-DT\\M_attr\\train"


for filename in os.listdir(dataset_root):
    if filename.endswith("_attr.txt"):
        file_path = os.path.join(dataset_root, filename)
        

        with open(file_path, 'r') as file:
            lines = file.readlines()

        image_folder = filename.replace("_attr.txt", "")
        image_folder_path = os.path.join(images_path, image_folder)

        for line in lines:
            values = line.strip().split(',')

            daylight = int(values[0])
            night = int(values[1] )
            fog = int(values[2])
            if daylight == 1:
                daylight_label = attribute_labels[0]
                destination_dir = os.path.join(dataset_root, daylight_label)
                os.makedirs(destination_dir, exist_ok=True)
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
                source_folder = images_path
                for image_filename in os.listdir(source_folder):
                    if image_filename.startswith(image_folder):
                        source_path = os.path.join(source_folder, image_filename)
                        shutil.copy(source_path, destination_dir)
                break

import os
import shutil

# 文件夹路径
all_labels_folder = "F:\\fsc_dataset\\UAV-DT\\dataset\\txt"
selected_images_folder = "F:\\fsc_dataset\\UAV-DT\\dataset\\weather\\val\\images\\daylight"
selected_labels_folder = "F:\\fsc_dataset\\UAV-DT\\dataset\\weather\\val\\txt\\daylight"

# 遍历新的图片文件夹
for image_filename in os.listdir(selected_images_folder):
    # 构建图片文件路径
    image_path = os.path.join(selected_images_folder, image_filename)

    # 构建标签文件路径
    label_filename = image_filename.replace(".jpg", ".txt")  # 假设标签文件是以.txt结尾的
    label_path = os.path.join(all_labels_folder, label_filename)

    # 如果标签文件存在，则进行复制
    if os.path.exists(label_path):
        # 构建目标标签文件夹路径
        destination_label_path = os.path.join(selected_labels_folder, label_filename)

        # 复制标签文件到新的标签文件夹
        shutil.copy(label_path, destination_label_path)
    else:
        print(f"Warning: No corresponding label file found for image {image_filename}. Skipping.")

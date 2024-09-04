import os
import shutil

all_labels_folder = "UAV-DT\\dataset\\txt"
selected_images_folder = "UAV-DT\\dataset\\weather\\val\\images\\daylight"
selected_labels_folder = "UAV-DT\\dataset\\weather\\val\\txt\\daylight"


for image_filename in os.listdir(selected_images_folder):

    image_path = os.path.join(selected_images_folder, image_filename)
    label_filename = image_filename.replace(".jpg", ".txt")
    label_path = os.path.join(all_labels_folder, label_filename)


    if os.path.exists(label_path):
        destination_label_path = os.path.join(selected_labels_folder, label_filename)
        shutil.copy(label_path, destination_label_path)
    else:
        print(f"Warning: No corresponding label file found for image {image_filename}. Skipping.")

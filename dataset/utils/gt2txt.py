import os

def parse_gt_folder(gt_folder):
    annotations = {}
    
    for filename in os.listdir(gt_folder):
        if filename.endswith('_gt_whole.txt'):
            file_path = os.path.join(gt_folder, filename)
            
            image_folder_name = filename.replace('_gt_whole.txt', '')
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        frame_index = int(parts[0])
                        bbox_left = float(parts[2])
                        bbox_top = float(parts[3])
                        bbox_width = float(parts[4])
                        bbox_height = float(parts[5])
                        object_category = int(parts[8])
                        
                        annotation = {
                            'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                            'category_id': object_category
                        }
                        
                        image_filename = f"{image_folder_name}_{frame_index:06d}.txt"
                        
                        if image_filename not in annotations:
                            annotations[image_filename] = []
                        annotations[image_filename].append(annotation)
    
    return annotations


def write_annotations_to_files(annotations, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for image_filename, annotation_list in annotations.items():
        output_file = os.path.join(output_folder, image_filename)
        
        with open(output_file, 'w') as f:
            for annotation in annotation_list:
                bbox_left, bbox_top, bbox_width, bbox_height = annotation['bbox']
                category_id = annotation['category_id']
                
                annotation_str = f"{bbox_left},{bbox_top},{bbox_width},{bbox_height},{category_id}\n"

                f.write(annotation_str)


gt_folder = 'UAV-DT\\UAV-benchmark-MOTD_v1.0\\GT'
output_folder = 'UAV-DT\\dataset\\txt'
annotations = parse_gt_folder(gt_folder)
write_annotations_to_files(annotations, output_folder)

import os

def parse_gt_folder(gt_folder):
    annotations = {}
    
    # 遍历文件夹中的每个 *_gt_whole.txt 文件
    for filename in os.listdir(gt_folder):
        if filename.endswith('_gt_whole.txt'):
            file_path = os.path.join(gt_folder, filename)
            
            # 获取当前文件名中的图片文件夹名称
            image_folder_name = filename.replace('_gt_whole.txt', '')
            
            # 读取每个文件中的标注信息
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
                        
                        # 构建标注信息，只包含bbox和category_id
                        annotation = {
                            'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                            'category_id': object_category
                        }
                        
                        # 构建图像文件名
                        image_filename = f"{image_folder_name}_{frame_index:06d}.txt"
                        
                        # 添加到对应图像文件名的字典中
                        if image_filename not in annotations:
                            annotations[image_filename] = []
                        annotations[image_filename].append(annotation)
    
    return annotations


def write_annotations_to_files(annotations, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历每个图像文件名及其对应的标注信息列表
    for image_filename, annotation_list in annotations.items():
        # 构建输出文件路径
        output_file = os.path.join(output_folder, image_filename)
        
        # 打开输出文件，写入标注信息
        with open(output_file, 'w') as f:
            for annotation in annotation_list:
                bbox_left, bbox_top, bbox_width, bbox_height = annotation['bbox']
                category_id = annotation['category_id']
                
                # 构建标注信息字符串
                annotation_str = f"{bbox_left},{bbox_top},{bbox_width},{bbox_height},{category_id}\n"
                
                # 写入标注信息到文件中
                f.write(annotation_str)


gt_folder = 'UAV-DT\\UAV-benchmark-MOTD_v1.0\\GT'
output_folder = 'UAV-DT\\dataset\\txt'
annotations = parse_gt_folder(gt_folder)
write_annotations_to_files(annotations, output_folder)

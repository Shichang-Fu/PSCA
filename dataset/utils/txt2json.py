import os
import json

from PIL import Image

coco_format_save_path='UAV-DT\\dataset\\'   
txt_format_classes_path='UAV-DT\\names.txt'
txt_format_annotation_path='UAV-DT\\dataset\\txt' 
img_pathDir='UAV-DT\\dataset\\images'

with open(txt_format_classes_path,'r') as fr:
    lines1=fr.readlines()

categories=[] 
for j,label in enumerate(lines1):
    label=label.strip()
    categories.append({'id':j+1,'name':label,'supercategory':'None'})

write_json_context=dict()  
write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2022, 'contributor': '0', 'date_created': '2022-07-8'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
write_json_context['categories']=categories
write_json_context['images']=[]
write_json_context['annotations']=[]

imageFileList=os.listdir(img_pathDir)                                       
for i,imageFile in enumerate(imageFileList):
    imagePath = os.path.join(img_pathDir,imageFile)                        
    image = Image.open(imagePath)                               
    W, H = image.size

    img_context={}                                                       
    #img_name=os.path.basename(imagePath)                              
    img_context['file_name']=imageFile
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2022-07-8'
    img_context['id']=i                                            
    img_context['license']=1
    img_context['color_url']=''
    img_context['flickr_url']=''
    write_json_context['images'].append(img_context)   
    
    txtFile=imageFile[:-4]+'.txt' 
    txtFilePath = os.path.join(txt_format_annotation_path, txtFile)
    if os.path.exists(txtFilePath):
        with open(txtFilePath, 'r') as fr:
            lines = fr.readlines()

        for j,line in enumerate(lines):
            bbox_dict = {}                                                      

            x,y,w,h,class_id=line.strip().split(',')                                      
            x,y,w,h,class_id =float(x), float(y), float(w), float(h) ,int(class_id)
            bbox_dict['id']=i*10000+j                            
            bbox_dict['image_id']=i
            bbox_dict['category_id']=class_id                        
            bbox_dict['iscrowd']=0
            bbox_dict['area']=h*w
            bbox_dict['bbox']=[x,y,w,h]
            bbox_dict['segmentation']=[]
            write_json_context['annotations'].append(bbox_dict)               

name = os.path.join(coco_format_save_path,"all"+ '.json')
with open(name,'w') as fw:                                                      
    json.dump(write_json_context,fw,indent=2)


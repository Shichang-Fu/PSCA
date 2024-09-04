import os
import json

from PIL import Image

coco_format_save_path='F:\\fsc_dataset\\UAV-DT\\dataset\\'   
txt_format_classes_path='F:\\fsc_dataset\\UAV-DT\\names.txt'
txt_format_annotation_path='F:\\fsc_dataset\\UAV-DT\\dataset\\txt' 
img_pathDir='F:\\fsc_dataset\\UAV-DT\\dataset\\images'

with open(txt_format_classes_path,'r') as fr:
    lines1=fr.readlines()

categories=[] 
for j,label in enumerate(lines1):
    label=label.strip()
    categories.append({'id':j+1,'name':label,'supercategory':'None'})

write_json_context=dict()                                                      #写入.json文件的大字典
write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2022, 'contributor': '0', 'date_created': '2022-07-8'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
write_json_context['categories']=categories
write_json_context['images']=[]
write_json_context['annotations']=[]

imageFileList=os.listdir(img_pathDir)                                           #遍历该文件夹下的所有文件，并将所有文件名添加到列表中
for i,imageFile in enumerate(imageFileList):
    imagePath = os.path.join(img_pathDir,imageFile)                             #获取图片的绝对路径
    image = Image.open(imagePath)                                               #读取图片，然后获取图片的宽和高
    W, H = image.size

    img_context={}                                                              #使用一个字典存储该图片信息
    #img_name=os.path.basename(imagePath)                                       #返回path最后的文件名。如果path以/或\结尾，那么就会返回空值
    img_context['file_name']=imageFile
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2022-07-8'
    img_context['id']=i                                                         #该图片的id
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
            bbox_dict = {}                                                          #将每一个bounding box信息存储在该字典中

            x,y,w,h,class_id=line.strip().split(',')                                          #获取每一个标注框的详细信息
            x,y,w,h,class_id =float(x), float(y), float(w), float(h) ,int(class_id)
            bbox_dict['id']=i*10000+j                                                         #bounding box的坐标信息
            bbox_dict['image_id']=i
            bbox_dict['category_id']=class_id                                              #注意目标类别要加一
            bbox_dict['iscrowd']=0
            bbox_dict['area']=h*w
            bbox_dict['bbox']=[x,y,w,h]
            bbox_dict['segmentation']=[]
            write_json_context['annotations'].append(bbox_dict)                               #将每一个由字典存储的bounding box信息添加到'annotations'列表中

name = os.path.join(coco_format_save_path,"all"+ '.json')
with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
    json.dump(write_json_context,fw,indent=2)


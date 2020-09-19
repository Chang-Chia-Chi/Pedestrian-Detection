import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt

date = datetime.date.today()
year, month, day = str(date).split('-')

# template of coco dataset
CoCo_Dataset = {
    "info":{
        "year": year, 
        "version": '1.0', 
        "description": 'WiderPerson dataset', 
        "contributor": 'WiderPerson', 
        "url": 'http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/', 
        "date_created": "{}-{}-{}".format(year, month, day)
    },
    "license":[
        {
        "id": 0, 
        "name": 'WiderPerson', 
        "url": "",
        }
    ],
    "images":[],
    "annotations":[],
    "categories":[{
        "id": 1, 
        "name": "person", 
        "supercategory": "",
}]
}

# Create coco data for pictures with crowd people
images_path = './crowd_people_pic/images'
annotations_path = './crowd_people_pic/annotations'

# create image list and sort it to ensure sequence
images = list(sorted(os.listdir(images_path)))
annotations = list(sorted(os.listdir(annotations_path)))

# Loading images and create dataset for "images" and "annotations"
image_id = 0
ann_id = 1
for image in images:
    img_path = os.path.join(images_path, image)
    img = plt.imread(img_path)

    img_id, _ = image.split('.')
    w, h, _ = img.shape
    img_dict = {
        "id": int(img_id),
        "license": CoCo_Dataset["license"][0]["id"],
        "coco_url": "",
        "flickr_url": "",
        "width": w,
        "height": h,
        "file_name": image,
        "date_captured": "{}-{}-{}".format(year, month, day)
    }
    CoCo_Dataset["images"].append(img_dict)

    ann_path = os.path.join(annotations_path, "{}.txt".format(img_id))
    with open(ann_path) as ann_file:
        lines = ann_file.readlines()

        # first one is number of people in the image
        for line in lines[1:]:
            class_label, x1, y1, x2, y2 = line.rstrip().split()
            if class_label == '1':
                
                x_min = float(x1)
                y_min = float(y1)
                box_width = float(x2) - x_min
                box_height = float(y2) - y_min
                box_area = box_width * box_height

                ann_dict = {
                    "id" : ann_id,
                    "category_id": int(class_label),
                    "iscrowd" : 0,
                    "segmentation": [],
                    "image_id" : int(img_id),
                    "area" : box_area,
                    "bbox" : [x_min, y_min, box_width, box_height]
                }

                CoCo_Dataset["annotations"].append(ann_dict)
                ann_id += 1
    image_id = int(img_id)

#-----------------------------------------------------------------------#
# Create coco data for pictures with few people
images_path = './few_people_pic/Images'
masks_path = './few_people_pic/Masks'

# create image list and sort it to ensure sequence
images = list(sorted(os.listdir(images_path)))
masks = list(sorted(os.listdir(masks_path)))

img_id = image_id
# Loading images and create dataset for "images" and "annotations"
for image in images:
    img_path = os.path.join(images_path, image)
    img = plt.imread(img_path)
    img_name = image.split('.')[0]
    w, h, _ = img.shape
    img_dict = {
        "id": img_id,
        "license": CoCo_Dataset["license"][0]["id"],
        "coco_url": "",
        "flickr_url": "",
        "width": w,
        "height": h,
        "file_name": image,
        "date_captured": "{}-{}-{}".format(year, month, day)
    }
    CoCo_Dataset["images"].append(img_dict)

    mask_path = os.path.join(masks_path, "{}_mask.png".format(img_name))
    mask_img = plt.imread(mask_path)
    obj_ids = np.unique(mask_img) # first one is background
    obj_ids = obj_ids[1:]

    for obj in obj_ids:
        pos = np.where(mask_img == obj)
        x_min = np.min(pos[1]).astype('float')
        x_max = np.max(pos[1]).astype('float')
        y_min = np.min(pos[0]).astype('float')
        y_max = np.max(pos[0]).astype('float')
        box_area = (x_max-x_min) * (y_max-y_min)

        ann_dict = {
            "id" : ann_id,
            "category_id": 1,
            "iscrowd" : 0,
            "segmentation": [],
            "image_id" : img_id,
            "area" : box_area,
            "bbox" : [x_min, y_min, (x_max-x_min), (y_max-y_min)]
        }

        CoCo_Dataset["annotations"].append(ann_dict)
        
        ann_id += 1
    img_id += 1

json.dump(CoCo_Dataset, open('train.json', 'w'))
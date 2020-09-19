# Pedestrian-Detection
This is my final project result of digital-image-processing course held by NCTU in 2020.
<p align="center">
  <img src="/performance/Final.gif" alt="Pedestrian Detection"></img>
</p>
    
The goal is to detect pedestrian in pictures or video, and count approximate number of people in the scene. Using pre-trained Fast-R-CNN model and transfer learning, the performance is pretty good with only 260 training pictures[[performance]](https://youtu.be/-qzeIVl4gJg). Because the problem scale is not big, **Google Colaboratory** is used for easy enviroment setting and free GPU.
## Dataset
Two dataset are used, one is from [pytorch official turtorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), the other is from [widerperson](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/) and randomly select 100 pictures. 
|Dataset             |Feature
| -------------------|-------
|**Pytorch tutorial**    |Only two~three people in the picture.
|**WiderPerson**    |Tens of people in the picture, which will highly enhance ability of ai to identify people in crowd. 

See [link](https://github.com/Chang-Chia-Chi/Pedestrian-Detection/tree/master/annotations) for training pictures and corresponding coco dataset json file.
## CoCo Dataset JSON Format [(reference)](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)
CoCo is abbreviation of **Common Objects in COntext**, quote from [cocodataset.org](https://cocodataset.org/#home): 

`COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image, 250,000 people with keypoints.`   
    
Coco has been used for so many projects because it's one of best image dataset in the world. Below will shortly introduce basic structure of coco dataset format for your own training data. You could find example `python` code in [link](https://github.com/Chang-Chia-Chi/Pedestrian-Detection/tree/master/annotations) which illustrate how to convert annotation or mask information to json file.   

### Structure and section
#### Structure   
```
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}
```   
#### section

**1. Info**: contains high level information about the dataset. 
```
"info":{   
    "year": int,      
    "version": str,     
    "description": str,   
    "contributor": str,   
    "url": str,   
    "date_created": datetime,   
}   
```   
**2. licence**: contains a list of image licenses that apply to images in the dataset.  
```
"license":[
        {
        "id": int, 
        "name": str,  
        "url": str,    
        }   
]   
```    
**3. images**: contains the complete list of images in your dataset.    
```
image{    
    "id": int,    
    "width": int,   
    "height": int,    
    "file_name": str,   
    "license": int,   
    "flickr_url": str,    
    "coco_url": str,    
    "date_captured": datetime,    
}
```   
**4. annotation**: contains a list of every individual object annotation from every image in the dataset.   
```
annotation{   
    "id": int,    
    "image_id": int,
    "category_id": int,
    "segmentation": [], (Fast-R-Cnn does not need this information)
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}
```   
This one is the most trickiest and important to understand, so a table collect purpose for each variable as below:    
|variable            |purpose
| -------------------|-------
|id                  |annotation id
|image_id            |corresponding image's id
|category_id         |which category of the object belongs to
|segmentation        |segmentation information for Mask-R-CNN
|area                |area for marked object in the picture, usually computed by height * width of box
|bbox                |x, y are position of left corner of box; width and height are box dimension
|iscrowd             |specifies whether the segmentation is for a single object or for a group/cluster of objects   

`P.S. Every object marked has one annotation. So it's one to one relationship between object and annotation id.`    

**5. catogories**: contains a list of categories (e.g. dog, person) and each of those belongs to a supercategory (e.g. animal, human).    
```
{   
    "id": int,    
    "name": str,    
    "supercategory": str,   
}   
```

# mxnet data IO   im2rec tutorial #



**this simple tutorial will introduce how to use im2rec for mx.image.ImageIter and ImageDetIter and how to use im2rec for COCO DataSet**
ok let's start but prepare your im2rec first
https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py

## im2rec ##
**step 1. you should make a .lst file** 
run in terminal
```
python im2rec.py --list testlst Dataset

--list     => make .lst file 
              you should set True when your folder didnt have any .lst file
              
test       => name of your .lst  name what you want 
DataSet    => your image dataset folder

check your folder path correct
```

**use .lst for imageIter**
```python

data_iter = mx.image.ImageIter(
    batch_size=4, 
    data_shape=(3,816, 1232),
    label_width=1,
    path_imglist='test.lst',
    path_root='DataSet')  


for data in data_iter:  
    d = data.data[0]  
    break  
  
img =nd.transpose(d,(0,2,3,1))  
print(img.shape)  
io.imshow(img[0].asnumpy().astype(np.uint8))  
io.show()

```
**step 2. make .rec file**
```
python im2rec.py testrec Dataset

testrec     => name of your .rec  name what you want
Dataset     => your image dataset folder
```
**use .rec for imageiter**
```python
train_iter = mx.image.ImageIter(  
    batch_size=32,  
   data_shape=(3, 816, 1232),  
   path_imgrec='data.rec',  
   path_imgidx='data.idx',  #help shuffle performance
   shuffle=True,  
   aug_list=[mx.image.HorizontalFlipAug(0.5)]  
)

  
train_iter.reset()  
for batch in train_iter:  
    x = batch.data[0]  
    break  
  
img =nd.transpose(x,(0,2,3,1))  
print(img.shape)  
io.imshow(img[0].asnumpy().astype(np.uint8))  
io.show()

```
**other feature**
```python
'--list' 
If this is set im2rec will create image list(s) by traversing root folder 
and output to .lst
Otherwise im2rec will read .lst and create a database at .rec

'--exts', default=['.jpeg', '.jpg', '.png'] 
list of acceptable image extensions.


'--train-ratio', default=1.0
Ratio of images to use for training.

'--test-ratio', default=0,  
Ratio of images to use for testing.

'--no-shuffle', dest='shuffle', action='store_false',  
If this is passed,im2rec will not randomize the image order in <prefix>.lst'


'--resize', type=int, default=0,  
resize the shorter edge of image to the newsize, original images will 
 be packed by default'

'--center-crop',  
specify whether to crop the center image to make it rectangular.

'--quality', default=95,  
JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9

'--num-thread',default=1,  
number of thread to use for encoding. order of images will be different


'--color', default=1, choices=[-1, 0, 1],  
specify the color mode of the loaded image.
 1: Loads a color image. Any transparency of image will be neglected. 
    It is the default flag 
 0: Loads image in grayscale mode.
-1:Loads image as such including alpha channel. 
    


```







# ImageDetIter #
## make own .lst for detection ##
**A  B  [extra header]  [(object0), (object1), ... (objectN)]**

1. you should write the .lst with your image information
Where A is the width of header (2 + length of extra header), B is the width of each object. Extra header is optional and used for inserting helper information such as (width, height). Each object is usually 5 or 6 numbers describing the object properties, for example: **[id, xmin, ymin, xmax, ymax, difficulty]** Putting all together, we have a `lst` file for object detection:

 ref: https://mxnet.incubator.apache.org/api/python/image/image.html
```python
with open('dataset.lst', 'w+') as f:  
    for i in range(3):  
        f.write(  
            str(i) + '\t' +  
            # idx  
            str(4) + '\t' + str(5) + '\t' +  
            # width of header and width of each object.  
            str(256) + '\t' + str(256) + '\t' +  
            # (width, height)  
            str(1) + '\t' +  
            # class  
            str((i / 10)) + '\t' + str((i / 10)) + '\t' + str(((i + 3) / 10)) + '\t' +str(((i + 3) / 10)) + '\t' +  
            # xmin, ymin, xmax, ymax  
           str(i) + '.jpg\n'
        )


```
you will see like this format in .lst
```
idx  A    B    [extra header]   [(object0), (object1), ... (objectN)]
0    4  5      256    256       1  0.1    0.1    0.3    0.3    0.jpg  
1    4  5      256    256       1  1.1    1.1    1.3    1.3    1.jpg  
2    4  5      256    256       1  2.1    2.1    2.3    2.3    2.jpg

 
```
**must **'\t'** between data information**

## make own dataset .rec for ImageDetIter ##
```
python im2rec.py --pack-label dataset.lst Dataset

--pack-label  => when you have label
dataset.lst     => name for .rec  type for your self in .lst
Dataset       => DataSet image folder
```

```python
import mxnet as mx  
import mxnet.ndarray as nd  
from skimage import io  
import numpy as np

train_iter = mx.image.ImageDetIter(  
   batch_size=3,  
   data_shape=(3, 256, 256),  
   path_imgrec='dataset.rec',  
   path_imgidx='dataset.idx',
   shuffle=True,  
   rand_crop=1,  
   min_object_covered=0.95,  
   max_attempts=200  
)  # you can aug your data in ImageDetIter
  
  
train_iter.reset()  
  
for batch in train_iter:  
    x = batch.data[0]  
    y = batch.label[0]  
    break  
  
print(y[0])  
  
  
img =nd.transpose(x,(0,2,3,1))  
print(img.shape)  
io.imshow(img[0].asnumpy().astype(np.uint8))  
io.show()

```

# How to use im2rec on COCO DataSet #
**1. download cococ dataset first!!!**
**2. make own .json or use COCO.json**  
**3. .json => .lst**  
**4. .lst  => .rec**
**5. .rec for ImageDetIter**

## how to load COCO DataSet from .json ##

```python	
import json  

with open('annotations/instances_train2017.json', 'r') as f:  
    DataSets = json.load(f)  
print(DataSets['annotations'][0])

>>{'segmentation': [[239.97, 260.24, 222.04, 270.49, 199.84, 253.41, 213.5, 227.79, 259.62, 200.46, 274.13, 202.17, 277.55, 210.71, 249.37, 253.41, 237.41, 264.51, 242.54, 261.95, 228.87, 271.34]],
 'area': 2765.1486500000005, 
 'iscrowd': 0, 
 'image_id': 558840, 
 'bbox': [199.84, 200.46, 77.71, 70.88], 
 'category_id': 58, 'id': 156}

```
**as you see, these annotations about image 558840 , boundingbox,class.... in detail you can check coco website**

## COCO.json => own.json ##

**make own dataset (class:bird) from COCO dataset**

```python
import json  
from mxnet import image  
from skimage import io  
import os  
  
  
  
## load COCO annotations  
with open('annotations/instances_train2017.json', 'r') as f:  
    DataSets = json.load(f)  
print(DataSets['annotations'][0])  
  
  
  
## save class and own dataset .json  
jsonName = 'ownset.json'  
directory = 'ownSet/'  
data = {}  
data['DataSet] = []  
with open(jsonName, 'w') as outfile:  
    if not os.path.exists(directory):  
        os.makedirs(directory)  
    for DataSet in DataSets['annotations']:  
        box = DataSet['bbox']  
        default_name = "000000000000"  
   img_id = str(DataSet['image_id'])  
        img_name = default_name[:len(default_name) - len(img_id)] + str(img_id) + '.jpg'  
   coco_name = 'train2017/' + img_name  
        if DataSet['category_id'] == 16:  #bird
  
            with open(coco_name, 'rb') as f:  
                img = image.imdecode(f.read())  
                height = img.shape[0]  
                width  = img.shape[1]  
                box[0] = box[0]/width  #normalize
                box[2] = box[2]/width  
                box[1] = box[1]/height  
                box[3] = box[3]/height  
            io.imsave(directory + img_name, img.asnumpy())  
            data['DataSet'].append({  
                'img_name': img_name,  
                'height': height,  
                'width': width,  
                'bbox': box,  
                'class':DataSet['category_id']  
            })  
    json.dump(data, outfile)  
print('save ok')  
  
  
  
with open(jsonName, 'r') as f:  
    Sets = json.load(f)  
print(Sets['DataSet'][0])

```
>{'img_name': '000000202273.jpg', 'height': 640, 'width': 480, 'bbox': [0.6530625000000001, 0.089296875, 0.33064583333333336, 0.075390625], 'class': 16}
>
**as you see this is your own dataset annotations**

## own data (.json) to .lst format ##
**1.How to use own own dataset.json and make own dataset .lst**
```python
import json  
import mxnet as mx  
from skimage import io  
  
jsonName = 'ownset.json'  
directory = 'ownSet/'  
with open(jsonName, 'r') as f:  
    DataSet = json.load(f)  
  
print(DataSet['DataSet'][0]['img_name'])  
  
img_idx = 0  
with open('ownSet.lst', 'w+') as f:  
    for Data in DataSet['DataSet']:  
  
        x_min = Data['bbox'][0]  
        y_min = Data['bbox'][1]  
        x_max = Data['bbox'][0]+ Data['bbox'][2]  
        y_max = Data['bbox'][1]+ Data['bbox'][3] 
        f.write(  
                str(img_idx) + '\t' +  # idx  
                str(4) + '\t' + str(5) + '\t' +  # width of header and width of each object.  
                str(int(Data['height'])) + '\t' + str(Data['width']) + '\t' +  # (width, height)  
                str(1) + '\t' +  # class  
                str(x_min) + '\t' + str(y_min) + '\t' + str(x_max) + '\t' + str(y_max) + '\t' +  # xmin, ymin, xmax, ymax  
                str(Data['img_name'])+'\n')  
        img_idx += 1



```
you will see your .lst like this format
 ![](https://github.com/leocvml/mxnet-im2rec_tutorial/blob/master/pic/lst.PNG)

 


## use im2rec for COCO dataSet ##
**this step just use  'bird' class (Previous) to show you, 
.lst  => .rec**

```
python im2rec.py --pack-label ownSet.lst ownSet

ownSet.lst => last step you make
ownSet     => image folder
```

## OK now we can use own dataSet ownSet.rec for ImageDetIter ##

```python
import mxnet as mx  
shape = 800  
train_iter = mx.image.ImageDetIter(  
    batch_size=32,  
   data_shape=(3, shape, shape),  
   path_imgrec='ownSet.rec',  
   path_imgidx='ownSet.idx',  
   shuffle=False,  
  
)  # you can aug your data in ImageDetIter  
  
import matplotlib.pyplot as plt  
def box_to_rect(box, color, linewidth=3):  
  
    box = box.asnumpy()  
    print((box[0], box[1]), box[2] - box[0], box[3]-box[1])  
    return plt.Rectangle(  
        (box[0], box[1]), box[2] - box[0], box[3]-box[1],  
   fill=False, edgecolor=color, linewidth=linewidth  
    )  
  
  
train_iter.reset()  
  
batch = train_iter.next()  
  
img, labels = batch.data[0], batch.label[0]  
  
print(labels.shape)  
  
img = img.transpose((0,2,3,1))  
img = img.clip(0,255).asnumpy()/255  
  
  
for i in range(32):  
    _, fig = plt.subplots()  
    plt.imshow(img[i])  
  
  
    rect = box_to_rect(labels[i][0][1:5]*shape,'red',2)  
    fig.add_patch(rect)  
    fig.axes.get_xaxis().set_visible(False)  
    fig.axes.get_yaxis().set_visible(False)  
    plt.show()



```
result
![](https://github.com/leocvml/mxnet-im2rec_tutorial/blob/master/pic/boundingboxresult.PNG)

## make very very example for multilabeling ##
will generate dataset.lst

```python
with open('dataset.lst', 'w+') as f:
    for i in range(12):
        f.write(
            str(i) + '\t' +
            # idx
            str(4) + '\t' + str(5) + '\t' +
            # width of header and width of each object.
            str(256) + '\t' + str(256) + '\t' +
            # (width, height)
            str(1) + '\t' +
            # class
            str((i / 15)) + '\t' + str((i / 15)) + '\t' + str(((i + 3) / 15)) + '\t' +str(((i + 3) / 15)) + '\t' +

            str(2) + '\t' +
            # class
            str((i / 50)) + '\t' + str((i / 50)) + '\t' + str(((i + 3) / 50)) + '\t' + str(((i + 3) / 50)) + '\t' +

            str(3) + '\t' +
            # class
            str((i / 100)) + '\t' + str((i / 100)) + '\t' + str(((i + 3) / 100)) + '\t' + str(((i + 3) / 100)) + '\t' +
            # xmin, ymin, xmax, ymax
           str(i) + '.jpg\n'
        )

```
check dataset.lst
![](https://github.com/leocvml/mxnet-im2rec_tutorial/blob/master/pic/multilabel.PNG)
## step2: use im2rec ##
```
python im2rec.py --pack-label dataset.lst data

```

## step3: use ImageDetIter show our multilabel ##
```python

import mxnet as mx
shape = 800
train_iter = mx.image.ImageDetIter(
    batch_size=32,
    data_shape=(3, shape, shape),
    path_imgrec='dataset.rec',
    path_imgidx='dataset.idx',
    shuffle=False,

)  # you can aug your data in ImageDetIter

import matplotlib.pyplot as plt
def box_to_rect(box, color, linewidth=3):

    box = box.asnumpy()
    print((box[0], box[1]), box[2] - box[0], box[3]-box[1])
    return plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth
    )


train_iter.reset()

batch = train_iter.next()

img, labels = batch.data[0], batch.label[0]

print(labels.shape)


img = img.transpose((0,2,3,1))
img = img.clip(0,255).asnumpy()/255


for i in range(12):
    _, fig = plt.subplots()
    plt.imshow(img[i])


    color_list = ['red','blue','black']
    for k in range(labels[i].shape[0]):  # how many object in your label
        rect = box_to_rect(labels[i][k][1:5]*shape,color_list[k],2)
        fig.add_patch(rect)
    fig.axes.get_xaxis().set_visible(False)
```
![](https://github.com/leocvml/mxnet-im2rec_tutorial/blob/master/multilabel_ressult.PNG)

**if you feel useful pls give a star!!!**

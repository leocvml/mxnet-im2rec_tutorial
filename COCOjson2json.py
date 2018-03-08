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
data['DataSet'] = []
with open(jsonName, 'w') as outfile:
    if not os.path.exists(directory):
        os.makedirs(directory)
    for DataSet in DataSets['annotations']:
        box = DataSet['bbox']
        default_name = "000000000000"
        img_id = str(DataSet['image_id'])
        img_name = default_name[:len(default_name) - len(img_id)] + str(img_id) + '.jpg'
        coco_name = 'train2017/' + img_name
        if DataSet['category_id'] == 16:

            with open(coco_name, 'rb') as f:
                img = image.imdecode(f.read())
                height = img.shape[0]
                width  = img.shape[1]
                box[0] = box[0]/width
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


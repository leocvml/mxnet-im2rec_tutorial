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


'''
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
'''



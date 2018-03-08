
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

HDFS = 'hdfs://hobot-bigdata/'


CIFAR10_ROOT = HDFS + 'user/xinyu.zhang/data/cifar10/'
CIFAR10 = {
    'train': CIFAR10_ROOT + 'train.rec',
    'val': CIFAR10_ROOT + 'val.rec'
}

IMAGENET_256_ROOT = HDFS + 'public_data/ilsvrc12_rec/'
IMAGENET_256 = {
    'train': IMAGENET_256_ROOT + 'train.rec',
    'val': IMAGENET_256_ROOT + 'val.rec'
}

MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'



VOC_CLASS_NAMES = 'aeroplane, bicycle, bird, boat, bottle, bus, \
                   car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                   person, pottedplant, sheep, sofa, train, tvmonitor'
VOC_CLASS_NAMES = [c.strip() for c in VOC_CLASS_NAMES.split(',')]
VOC_ROOT = HDFS + 'user/xinyu.zhang/data/VOC/'
VOC = {
    'label_width': 350,
    'mean_pixels': [123, 117, 104],
    'train_path': VOC_ROOT + 'train.rec',
    'train_list': VOC_ROOT + 'train.lst',
    'val_path': VOC_ROOT + 'val.rec',
    'val_list': VOC_ROOT + 'val.lst'
}


RESCALE_GRAD = {
    'divide_batch_size': 'divide_batch_size',
    'divide_num_devs': 'divide_num_devs'
}

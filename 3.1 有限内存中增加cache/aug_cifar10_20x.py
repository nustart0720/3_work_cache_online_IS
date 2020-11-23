import os

import imageio
import imgaug as ia
from imgaug import augmenters as iaa

# augment train images in cifar10 to 20 times, from 50,000 in total to 1000,000 in total
cifar10_train_path = '/data/cifar10/train'
pic_id = 5000

def aug_and_write_20times(file_path):
    global pic_id
    image = imageio.imread(file_path)

    for i in range(1,20):
        pic_id += 1

        seq = iaa.Sequential([
            iaa.Affine(rotate=(-20, 20)),
            iaa.Crop(percent=(0, 0.2))], random_order=True)
        image_aug = seq(image=image)

        save_file_path = os.path.join(os.path.dirname(file_path),str(pic_id)+'.png')
        imageio.imwrite(save_file_path,image_aug)



# in every catagory images number from 5001 to 100,000
for catag in os.listdir(cifar10_train_path):
    dirname = os.path.join(cifar10_train_path,catag)
    if os.path.isdir(dirname):
        pic_id = 5000
        for f in os.listdir(dirname):
            full_path = os.path.join(dirname,f)
            aug_and_write_20times(full_path)
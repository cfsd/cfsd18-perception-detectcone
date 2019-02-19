import numpy as np
import os
import cv2
import random
import glob

patch_size = 64
radius = 32

def augmentation(img):
    random_rate = random.random()
    if random_rate > 0.8:
        R = int(patch_size*random_rate/2)
        shift = int(abs(R-radius+1)/2)
        x_shift = random.randint(-shift,shift)
        y_shift = random.randint(-shift,shift)
        img = img[radius-R+y_shift:radius+R+y_shift, radius-R+x_shift:radius+R+x_shift]
        img = img[radius-R:radius+R, radius-R:radius+R]
        img = cv2.resize(img, (patch_size, patch_size))
    if random.random() > 0.5:
        img = img[:, ::-1]
    return img

def split_dataset(data_paths):
    copies = 5
    for data_path in data_paths:
        for i in range(4):
            for img_path in glob.glob(os.path.join(data_path, str(i), '*.png')):
                img = cv2.imread(img_path)
                if i in [3]:
                    if random.random() < 7/(3*copies+7):
                        save_path = os.path.join('data/train', str(i))
                        for k in range(copies):
                            img = augmentation(img)
                            num = 0
                            while os.path.exists(os.path.join(save_path, str(num)+'.png')):
                                num += 1
                            cv2.imwrite(os.path.join(save_path, str(num)+'.png'),img)
                    else:
                        save_path = os.path.join('data/test', str(i))
                        num = 0
                        while os.path.exists(os.path.join(save_path, str(num)+'.png')):
                            num += 1
                        cv2.imwrite(os.path.join(save_path, str(num)+'.png'),img)
                else:
                    if random.random() < 0.7:
                        save_path = os.path.join('data/train', str(i))
                        img = augmentation(img)
                    else:
                        save_path = os.path.join('data/test', str(i))
                    num = 0
                    while os.path.exists(os.path.join(save_path, str(num)+'.png')):
                        num += 1
                    cv2.imwrite(os.path.join(save_path, str(num)+'.png'),img)

#split_dataset(['/media/weiming/46823A5A823A4F253/data/2018-07-26_174108/annotations-hard',
#    '/media/weiming/46823A5A823A4F253/data/2018-08-10_133910/annotations-hard'])
split_dataset(['/home/ruguang/docker-CFSD/local-replay/perception-replay/cfsd-perception-detectcone/cnn/data/train'])

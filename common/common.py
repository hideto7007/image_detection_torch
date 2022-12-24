import os.path as osp
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 機械学習モジュール
from torchvision import models, transforms

# 自作モジュール
from config import Config
from transform.transform import ImageTransform

# インスタンス化
# pathは既にconfig内で定義済み
config = Config()
transform = ImageTransform(config.size, config.mean, config.std)



class Common:
    def __init__(self):
        self.path_list = []

    # アリとハチの画像へのファイルパスのリスト作成する

    def make_datapath_list(self, phase, path):
        rootpath = config.path
        target_path = osp.join(rootpath+phase+'/*/*.jpg')
        
        self.path_list = []
        
        for path in glob.glob(target_path):
            self.path_list.append(path)
            
        return self.path_list
    
    def img_show(self):
        imgae_file_path = config.path + 'train/ants/000001_r.jpg'
        img = Image.open(imgae_file_path)
        
        plt.imshow(img)
        plt.show()
        
        img_transformed = transform(img, phase="train")
        img_transformed = img_transformed.numpy().transpose((1, 2, 0))
        img_transformed = np.clip(img_transformed, 0, 1)
        plt.imshow(img_transformed)
        plt.show()
        


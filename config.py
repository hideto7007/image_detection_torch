

class Config:
    def __init__(self):
        self.num_epochs = 30
        self.path = "./img_data/"
        self.flg = False
        self.cuda = 0
        # ミニバッチのサイズを指定
        self.batch_size = 20
        # GPU設定判別
        if self.cuda:
            self.size = 2242
        else:
            self.size = 100
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        # vgg16, resnet, efficientNet
        self.model = "efficientNet"
        self.fine_flag = 1
        self.out_feature = 2

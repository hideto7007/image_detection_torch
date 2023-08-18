



# 機械学習モジュール
import torch

# 自作モジュール
from config import Config



# インスタンス化
# pathは既にconfig内で定義済み
config = Config()


# モデル保存先

dirs = ["before_model", "after_model"]

if config.model == "vgg16":
    load_path = "./models/" + dirs[1] + "/vgg16_best_models.pth"
    model_params = torch.load(load_path)
elif config.model == "resnet":
    load_path = "./models/" + dirs[1] + "/resnet_best_models.pth"
    model_params = torch.load(load_path)
elif config.model == "efficientNet":
    load_path = "./models/" + dirs[1] + "/efficientNet_best_models.pth"
    model_params = torch.load(load_path)
    
    
print(model_params)



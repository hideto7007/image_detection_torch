# 外部モジュール
import os.path as osp
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# 機械学習モジュール
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

# 自作モジュール
from config import Config
from transform.transform import ImageTransform
from dataset.dataset import HymenopteraDataset
from common.common import Common
from models.models import learned_models, optimizer


# 乱シード設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benckmark = False

# 画像を表示判別
image_flag = 0


# インスタンス化
# pathは既にconfig内で定義済み
config = Config()
common = Common()
transform = ImageTransform(config.size, config.mean, config.std)

if image_flag:
    common.img_show()

# 画像パスを作成
train_list = common.make_datapath_list(phase="train", path=config.path)
val_list = common.make_datapath_list(phase="val", path=config.path)

# Datasetを作成
train_dataset = HymenopteraDataset(
    file_list=train_list, transform=transform, phase='train')
val_dataset = HymenopteraDataset(
    file_list=val_list, transform=transform, phase='val')

# DataLoaderを作成
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作確認
batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
inputs, labels = next(batch_iterator)  # 1番目の要素を取り出す

print(batch_iterator)
print(inputs.size())
print(labels)


# バッチサイズは全て一律の値でないと、モデル学習時に値が一致してなくエラーとなる
# バッチサイズが一致してるか確認
for i in train_dataloader:
    print(i[1].size())


# 損失関数の設定
criterion = nn.CrossEntropyLoss()


# モデル保存先

dirs = ["before_model", "after_model"]

if config.model == "vgg16":
    save_path = "./models/" + dirs[1] + "/vgg16_best_models.pth"
elif config.model == "resnet":
    save_path = "./models/" + dirs[1] + "/resnet_best_models.pth"
elif config.model == "efficientNet":
    save_path = "./models/" + dirs[1] + "/efficientNet_best_models.pth"


# 使用するモデル設定かつパラメータ一覧取得

prame_1, prame_2, prame_3, prame, net = None, None, None, None, None

if config.model == "vgg16" and config.fine_flag:
    net, prame_1, prame_2, prame_3 = learned_models(
        config.model, config.fine_flag, config.out_feature, models, nn)
elif config.model == "resnet" and config.fine_flag:
    net, prame_1, prame_2, prame_3 = learned_models(
        config.model, config.fine_flag, config.out_feature, models, nn)
elif config.model == "efficientNet" and config.fine_flag:
    net, prame_1, prame_2 = learned_models(
        config.model, config.fine_flag, config.out_feature, models, nn)
elif config.model == "vgg16":
    net, prame = learned_models(
        config.model, config.fine_flag, config.out_feature, models, nn)
elif config.model == "resnet":
    net, prame = learned_models(
        config.model, config.fine_flag, config.out_feature, models, nn)
elif config.model == "efficientNet":
    net, prame = learned_models(
        config.model, config.fine_flag, config.out_feature, models, nn)


# 設定したパラメータ取得
if config.fine_flag:
    optimizer = optimizer(net, optim, config.model,
                          config.fine_flag, prame_1, prame_2, prame_3, prame)
else:
    optimizer = optimizer(net, optim, config.model,
                          config.fine_flag, prame_1, prame_2, prame_3, prame)


# モデルを学習させる関数を作成

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # 初期設定
    # GPUが使えるかを確認
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("使用デバイス：", device)

    # ネットワークをGPUへ
    # net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    all_epoch_acc = []  # 正解率リストに保存
    count = 0

    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPUが使えるならGPUにデータを送る
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 更新前のlossを保持
                    before_epoch_loss = epoch_loss

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)
            # 更新前のepochごとのlossと正解率を表示
            before_epoch_loss = before_epoch_loss / \
                len(dataloaders_dict[phase].dataset)
            # 検証の正解率を保持
            if phase == "val":
                all_epoch_acc.append(format(epoch_acc, '.4f'))
                count += 1

            if count-1 != 0 and phase == "val":
                # 一番良いモデルを保存
                if format(epoch_acc, '.4f') > max(all_epoch_acc[:-1]):

                    torch.save(net.state_dict(), save_path)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    print("max_num: ", max(all_epoch_acc))


print("学習モデル:", config.model)
train_model(net, dataloaders_dict, criterion,
            optimizer, num_epochs=config.num_epochs)

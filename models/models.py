

def learned_models(model, fine_flag, out_feature, models, nn):
    """学習済みモデル選定"""
    
    use_pretrained = True  # 学習済みのパラメータを使用

    if model == "vgg16":
        # VGG-16モデルのインスタンスを生成  
        net = models.vgg16(pretrained=use_pretrained)
        net.classifier[6] = nn.Linear(in_features=4096, out_features=out_feature)
        # パラメータ名
        if fine_flag:
            update_param_names_1 = ["features"]
            update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
            update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]
        else:
            update_param_names = ["classifier.6.weight", "classifier.6.bias"]
    elif model == "resnet":
        # ResNet50モデルのインスタンスを生成
        net = models.resnet50(pretrained=use_pretrained)
        net.fc = nn.Linear(net.fc.in_features, out_feature)
        # パラメータ名
        if fine_flag:
            update_param_names_1 = ["conv1.weight", "bn1.weight", "bn1.bias"]
            update_param_names_2 = ["layer"]
            update_param_names_3 = ["fc.weight", "fc.bias"]
        else:
            update_param_names = ["fc.weight", "fc.bias"]
    elif model == "efficientNet":
        # EfficientNetモデルのインスタンスを生成
        net = models.efficientnet_b7(pretrained=use_pretrained)
        net.classifier[1] = nn.Linear(in_features=2560, out_features=out_feature, bias=True)
        # パラメータ名
        if fine_flag:
            update_param_names_1 = ["features"]
            update_param_names_2 = ["classifier.1.weight", "classifier.1.bias"]
        else:
            update_param_names = ["classifier.1.weight", "classifier.1.bias"]
    # print(net.classifier[1])

    # 訓練モードに設定
    net.train()

    print('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')
    
    if model == "vgg16" and fine_flag:
        return net, update_param_names_1, update_param_names_2, update_param_names_3
    elif model == "resnet" and fine_flag:
        return net, update_param_names_1, update_param_names_2, update_param_names_3
    elif model == "efficientNet" and fine_flag:
        return net, update_param_names_1, update_param_names_2
    elif model == "vgg16":
        return net, update_param_names
    elif model == "resnet":
        return net, update_param_names
    elif model == "efficientNet":
        return net, update_param_names
    
    
def optimizer(net, optim, model, fine_flag, update_param_names_1, update_param_names_2, update_param_names_3, update_param_names):
    """パラメータ最適化設定"""
    
    # 転移学習で学習させるパラメータを、変数params_to_updateに格納する
    params_to_update = []
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []
    
    # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
    if fine_flag:
        for name, param in net.named_parameters():
            if update_param_names_1[0] in name and (model == "vgg16" or model == "efficientNet"):
                param.requires_grad = True
                params_to_update_1.append(param)
                # print("params_to_update_1に格納", name)

            elif name in update_param_names_1 and (model == "resnet"):
                param.requires_grad = True
                params_to_update_1.append(param)
                # print("params_to_update_1に格納", name)

            elif name in update_param_names_2 and (model == "vgg16" or model == "efficientNet"):
                param.requires_grad = True
                params_to_update_2.append(param)
                # print("params_to_update_2に格納", name)

            elif update_param_names_2[0] in name and (model == "resnet"):
                param.requires_grad = True
                params_to_update_2.append(param)
                # print("params_to_update_2に格納", name)

            elif name in update_param_names_3 and (model == "vgg16" or model == "resnet"):
                param.requires_grad = True
                params_to_update_3.append(param)
                # print("params_to_update_3に格納", name)
            else:
                param.requires_grad = False

    else:
        for name, param in net.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
                # print(name)
            else:
                param.requires_grad = False
                
                
    if fine_flag:
        if model == "vgg16":
            params = [
                {'params': params_to_update_1, 'lr': 1e-4},
                {'params': params_to_update_2, 'lr': 5e-4},
                {'params': params_to_update_3, 'lr': 1e-3}
            ]

        elif model == "resnet":
            params = [
                {'params': params_to_update_1, 'lr': 1e-4},
                {'params': params_to_update_2, 'lr': 5e-4},
                {'params': params_to_update_3, 'lr': 1e-3}
            ]
        elif model == "efficientNet":
            params = [
                {'params': params_to_update_1, 'lr': 1e-4},
                {'params': params_to_update_2, 'lr': 1e-3}
            ]
          
    if fine_flag:  
        # optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
        optimizer = optim.Adam(params, amsgrad=False)
    else:
        # optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        optimizer = optim.Adam(params_to_update, amsgrad=False)
        
    return optimizer
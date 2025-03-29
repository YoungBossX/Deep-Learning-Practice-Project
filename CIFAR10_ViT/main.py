# -*-coding: utf-8 -*-
# @Time    : 2025/2/23 13:39
# @Author  : XCC
# @File    : main.py
# @Software: PyCharm

import argparse
import os
import torch.optim
from torch import nn
from utils import init_logger, set_seed
from data_loader import CIFAR10Dataset, transform_train1, transform_eval1, transform_train2, transform_eval2, transform_train3, transform_eval3
from trainer import Trainer
from models import CNN, AlexNet, VGG16, GoogLeNet, ResNet18, ViT

# itos = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
# stoi = dict((v, k) for k, v in itos.items())

def main(args, model_name):
    logger = init_logger(args, __name__, model_name)  # 配置日志器
    set_seed(args)  # 设置随机种子

    # 根据不同的模型选择不同的图像增强方法
    transform_train = None
    transform_eval = None
    if model_name == 'CNN':
        transform_train = transform_train1
        transform_eval = transform_eval1
    elif model_name == 'MyAlexNet':
        transform_train = transform_train2
        transform_eval = transform_eval2
    elif model_name in ['MyVGG16', 'MyGoogLeNet', 'MyResNet18', 'MyViT', 'ViT']:
        transform_train = transform_train3
        transform_eval = transform_eval3


    # 加载数据
    logger.info("******  Loading dataset...  ******")
    train_dataset = CIFAR10Dataset(data_dir=args.data_dir, transform=transform_train, mode='train')  # 加载训练集
    logger.info("Num of train_dataset examples: {}".format(len(train_dataset)))

    dev_dataset = CIFAR10Dataset(data_dir=args.data_dir, transform=transform_eval, mode='val')  # 加载验证集
    logger.info("Num of val_dataset examples: {}".format(len(dev_dataset)))

    test_dataset = CIFAR10Dataset(data_dir=args.data_dir, transform=transform_eval, mode='test')  # 加载测试集
    logger.info("Num of test_dataset examples: {}".format(len(test_dataset)))
    logger.info("******  The dataset is loaded!  ******\n")


    # 构建模型
    logger.info("******  Start building the model...  ******")
    model = None
    if model_name == 'CNN':
        model = CNN.CNN()
    elif model_name == 'MyAlexNet':
        model = AlexNet.AlexNet()
    elif model_name == 'MyVGG16':
        model = VGG16.VGG16()
    elif model_name == 'MyGoogLeNet':
        model = GoogLeNet.GoogLeNet()
    elif model_name == 'MyResNet18':
        model = ResNet18.ResNet18(ResNet18.BasicBlock)
    elif model_name == 'ViT':
        model = ViT.vit_base_patch16_224(num_classes=10)


    # 使用预训练模型
    # elif model_name == 'ViT':
    #     model_name_pre = "../../huggingface/google_vit-base-patch16-224"
    #     model = ViTForImageClassification.from_pretrained(model_name_pre, num_labels=10, ignore_mismatched_sizes=True,
    #                                                       id2label=itos, label2id=stoi)
    #     # 只训练最后的分类层
    #     for name, param in model.named_parameters():
    #         param.requires_grad = False
    #         if 'classifier' in name:
    #             param.requires_grad = True
    #     params = filter(lambda p: p.requires_grad, model.parameters())

    logger.info("The structure of the models:\n{}".format(model))
    logger.info("******  The model is built!  ******\n")

    Resume = False
    # Resume = False
    if Resume:
        weights_path = args.model_weights
        weights = torch.load(os.path.join(args.model_weights, model_name+'.pt'), map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(weights)


    # 定义损失函数和优化器
    # if model_name == 'ViT':
    #     optimizer = torch.optim.Adam(params, lr=args.learning_rate)  # 使用Adam()优化器（只传入需要更新的参数）
    # else:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 使用Adam()优化器
    # loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 使用Adam()优化器
    loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 训练
    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset, model, optimizer, loss_func, model_name)

    # 训练模型
    if args.do_train:
        trainer.train()

    # 评估模型
    if args.do_eval:
        trainer.load_model()
        accuracy = trainer.evaluate(mode='test')
        logger.info("Test Accuracy: {:.3f}".format(accuracy))


if __name__ == '__main__':

    # 初始化参数解析器
    parser = argparse.ArgumentParser()

    # 各种文件路径参数
    parser.add_argument("--data_dir", default="E:\数据集\datasets", type=str, help="The input data dir")  # 数据文件夹
    parser.add_argument("--model_weights", default="./weights", type=str, help="Path for saving models")  # 模型保存路径
    parser.add_argument("--logs_dir", default="./logs", type=str, help="Path for saving log")  # 数据文件夹

    # 辅助模块中参数
    parser.add_argument("--seed", type=int, default=6, help="random seed for initialization")  # 随机种子

    # 模型训练超参数
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")  # 总训练轮数
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and testing")  # 批次大小
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="The initial learning rate")  # 学习率

    # 控制训练/测试参数
    parser.add_argument("--do_train", default=True, type=bool, help="Whether to run training.")  # 是否训练
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to run eval on the test set.")

    # 定义参数
    args = parser.parse_args()


    # 选择使用哪个模型：CNN/AlexNet/VGG16/GoogLeNet/ResNet18/ViT
    model_name = 'ViT'

    # 调用主函数
    main(args, model_name)
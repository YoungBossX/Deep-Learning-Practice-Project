# -*-coding: utf-8 -*-
# @Time    : 2025/2/22 14:29
# @Author  : XCC
# @File    : main.py
# @Software: PyCharm

import argparse
import torch
from torch import nn
from utils import init_logger, set_seed
from data_loader import load_dataset
from trainer import Trainer
from model import Net

def main(args):
    logger = init_logger(args, __name__)  # 配置日志器
    set_seed(args)  # 设置随机种子

    # 加载数据
    logger.info("******  Loading dataset...  ******")
    train_dataset = load_dataset(args, mode="train")  # 加载训练数据
    test_dataset = load_dataset(args, mode="test")  # 加载测试数据
    logger.info("******  The dataset is loaded!  ******\n")

    # 构建模型
    logger.info("******  Start building the model...  ******")
    net = Net()
    logger.info("The structure of the model:\n{}".format(net))
    logger.info("******  The model is built!  ******\n")

    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # 使用Adam()优化器
    loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 训练
    trainer = Trainer(args, train_dataset, test_dataset, net, optimizer, loss_func)

    # 训练模型
    if args.do_train:
        trainer.train()
        trainer.save_model()

    # 评估模型
    if args.do_eval:
        trainer.load_model()
        accuracy = trainer.test()
        logger.info("Test Accuracy: {:.3f}".format(accuracy))

if __name__ == '__main__':
    # 初始化参数解析器
    parser = argparse.ArgumentParser()
    # 各种文件路径参数
    parser.add_argument("--data_path", default="./datasets", type=str, help="The data path.")
    parser.add_argument("--model_dir", default="./models", type=str, help="Path for saving model.")
    parser.add_argument("--logs_dir", default="./logs", type=str, help="Path for saving log.")
    # 训练参数
    parser.add_argument("--seed", default=7, type=int, help="Random seed.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="The number of training epochs.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--do_train", default=True, type=bool, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to run eval on the test set.")
    args = parser.parse_args()

    main(args)
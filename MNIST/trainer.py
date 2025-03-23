# -*-coding: utf-8 -*-
# @Time    : 2025/2/22 14:30
# @Author  : XCC
# @File    : trainer.py
# @Software: PyCharm

import os
import torch
from torch.utils.data import DataLoader
from utils import init_logger

class Trainer(object):
    def __init__(self, args, train_dataset=None, test_dataset=None, model=None, optimizer=None, loss_func=None):
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.logger = init_logger(args, __name__)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self):
        train_data_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.logger.info("******  Running training  ******")
        self.logger.info("  Num examples = %d", len(self.train_dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        # 最优模型
        max_accuracy = 0

        for epoch in range(self.args.num_train_epochs):
            self.logger.info("\n-------------------- Epoch: {} --------------------".format(epoch+1))
            for step, batch in enumerate(train_data_loader):
                self.model.train()  # 训练状态
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                output = self.model(x)
                loss = self.loss_func(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                steps = step + 1
                if steps % 200 == 0:  # 每200批进行一次测试
                    accuracy = self.test()
                    self.logger.info("Epoch: {} | Step: {} | train loss: {:.4f} | test accuracy: {:.3f}"
                                 .format(epoch+1, step+1, loss .data.cpu().numpy(), accuracy))

    def test(self):
        test_data_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False)
        result = 0
        total = 0
        with torch.no_grad():
            for batch in test_data_loader:
                self.model.eval()
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                output = self.model(x)
                y_pred = torch.argmax(output, 1).cpu().numpy()
                result += (y.cpu().numpy() == y_pred).sum()
                total += y.size(0)
        accuracy = result / total
        return accuracy

    def save_model(self):
        # 判断是否存在文件夹，如果不存在则新建
        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        # 保存模型参数
        self.logger.info("******  Start Saving Model...  ******")
        torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, 'model.pt'))
        self.logger.info("Saving model checkpoint to {}\n".format(os.path.join(self.args.model_dir, 'model.pt')))

    def load_model(self):
        # 检查模型是否存在
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        # 加载模型参数
        self.logger.info("******  Start Loading Model...  ******")
        self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, 'model.pt'), weights_only=True))
        self.model.to(self.device)
        self.logger.info("******  The Model is Loaded!  ******")
# -*-coding: utf-8 -*-
# @Time    : 2025/2/23 13:40
# @Author  : XCC
# @File    : trainer.py
# @Software: PyCharm

import os
import torch

from torch.utils.data import DataLoader

from utils import init_logger

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, model=None, optimizer=None, loss_func=None, model_name=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.model_name = model_name
        self.logger = init_logger(self.args, __name__, model_name)

        # GPU or CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):

        # Train
        self.logger.info("******  Running training  ******")
        self.logger.info("  Num examples = %d", len(self.train_dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        # 打包数据
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

        # 最优模型
        max_accuracy = 0
        max_epoch = 0

        for epoch in range(self.args.num_train_epochs):
            self.logger.info("\n-------------------- Epoch: {} --------------------".format(epoch+1))
            for step, batch in enumerate(train_dataloader):
                self.model.train()  # 训练状态
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                output = self.model(x)
                if self.model_name == 'ViT':
                    output = output.logits
                loss = self.loss_func(output, y.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 50 == 49:  # 每50批进行一次测试
                    accuracy = self.evaluate(mode='dev')
                    self.logger.info("Epoch: {} | Step: {} | train loss: {:.4f} | dev accuracy: {:.3f}"
                                 .format(epoch+1, step+1, loss.data.cpu().numpy(), accuracy))

            # 每一轮判断是否要保存一次模型
            accuracy = self.evaluate(mode='dev')
            if accuracy > max_accuracy:
                self.save_model()
                max_accuracy = accuracy
                max_epoch = epoch + 1
            else:
                self.logger.info("The models is not updated!")

            self.logger.info(f"The best models is Epoch {max_epoch}!\n")

    def evaluate(self, mode):
        # 根据不同模式进行评估
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        # 打包数据
        eval_dataloader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=False)

        # Eval
        correct = total = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                self.model.eval()  # 评估状态
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                output = self.model(x)
                if self.model_name == 'ViT':
                    output = output.logits
                y_pred = torch.argmax(output, 1).cpu().numpy()
                correct += (y.cpu().numpy() == y_pred).sum()
                total += y.size(0)
            accuracy = correct/total
            return accuracy


    def save_model(self):
        # 判断是否存在文件夹，如果不存在则新建
        if not os.path.exists(self.args.checkpoints_dir):
            os.mkdir(self.args.checkpoints_dir)

        # 保存模型参数
        self.logger.info("******  Start Saving Model...  ******")
        torch.save(self.model.state_dict(), os.path.join(self.args.checkpoints_dir, self.model_name+'.pt'))
        self.logger.info("Saving models checkpoint to {}".format(os.path.join(self.args.checkpoints_dir, self.model_name+'.pt')))


    def load_model(self):
        # 检查模型是否存在
        if not os.path.exists(self.args.checkpoints_dir):
            raise Exception("Model doesn't exists! Train first!")

        self.logger.info("******  Start Loading Model...  ******")
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints_dir, self.model_name+'.pt')))
        self.model.to(self.device)
        self.logger.info("******  The Model is Loaded!  ******")
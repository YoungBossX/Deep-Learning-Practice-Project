# -*-coding: utf-8 -*-
# @Time    : 2025/2/23 13:39
# @Author  : XCC
# @File    : utils.py
# @Software: PyCharm

import os
import random
import logging

import torch
import numpy as np


# 配置日志器（格式、时间）
def init_logger(args, logger_name, model_name):
    # Create logs directory if it doesn't exist
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # 创建日志器对象
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S') # 月/日/年 时:分:秒

    # 写入文件
    file_handler = logging.FileHandler(os.path.join(args.logs_dir, model_name+'.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    if not logger.handlers:
        # 控制台显示
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 设置随机种子
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
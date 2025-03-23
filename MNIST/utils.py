# -*-coding: utf-8 -*-
# @Time    : 2025/2/22 14:29
# @Author  : XCC
# @File    : utils.py
# @Software: PyCharm

import os
import random
import logging
import torch
import numpy as np
from argparse import Namespace  # 用于模拟 args 参数

# 配置日志器（格式、时间）
def init_logger(args, logger_name):
    # 创建日志器对象
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    # 写入文件
    file_handler = logging.FileHandler(os.path.join(args.logs_dir, 'info.txt'))
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

if __name__ == '__main__':
    # 创建模拟的 args 参数对象
    args = Namespace(
        logs_dir=os.path.join(os.getcwd(), "logs")  # 指定日志存放目录（这里用当前工作目录下的 logs 文件夹）
    )

    # 确保日志目录存在
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # 初始化日志器
    logger = init_logger(args, "TEST_Logger")

    # 测试日志输出
    logger.debug("这是一个 DEBUG 级别的消息（应写入文件但不在控制台显示）")
    logger.info("这是一个 INFO 级别的消息（应同时写入文件和控制台）")
    logger.warning("这是一个 WARNING 级别的消息")
    logger.error("这是一个 ERROR 级别的消息")

    print("\n测试完成，请检查以下内容：")
    print(f"1. 日志文件是否生成：{os.path.join(args.logs_dir, 'test.log')}")
    print("2. 控制台是否显示 INFO 及以上级别的消息")
    print("3. 日志文件内容是否符合格式（包含时间、级别、名称和消息）")
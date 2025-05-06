import os
import random
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可重现
def setup_seed(seed):
    """
    设置随机种子，使得实验可重复。
    Args:
        seed (int): 随机种子
    """
    torch.manual_seed(seed)  # 设置CPU随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python的随机模块的种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次计算结果相同
    torch.backends.cudnn.benchmark = False  # 避免在不同的环境下优化


# 创建一个日志记录器
def make_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 如果日志目录不存在，创建目录

    log_file = os.path.join(log_dir, "training.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_dir


# 保存模型
def save_model(model, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# 加载模型
def load_model(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: {model_path} does not exist!")


# 绘制训练过程中的损失曲线
def plot_loss_curve(losses, save_path=None):
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
    else:
        plt.show()

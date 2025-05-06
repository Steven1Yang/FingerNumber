import torch.nn as nn
import torch.optim as optim


def get_criterion(loss_type="CrossEntropyLoss"):
    if loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_type == "MSELoss":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def get_optimizer(model, optimizer_type="Adam", lr=0.001, weight_decay=1e-4, scheduler_type="StepLR"):
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    if scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return optimizer, scheduler


if __name__ == "__main__":
    # 示例：测试损失函数和优化器
    from Model import GestureRecognitionModel  # 你的模型文件（model.py）

    model = GestureRecognitionModel()

    criterion = get_criterion()  # 获取损失函数
    optimizer, scheduler = get_optimizer(model)  # 获取优化器和调度器

    print("损失函数:", criterion)
    print("优化器:", optimizer)
    print("学习率调度器:", scheduler)
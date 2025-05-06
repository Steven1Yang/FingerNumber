import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import utils.my_utils as utils  # 假设你有一个自定义的 utils 工具
from Model import GestureRecognitionModel  # 导入你的模型
from DataProcess import fingergraph  # 导入你的数据集类
from torchvision import transforms
from Optimizer import get_criterion, get_optimizer
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging
import os
# 关闭 oneDNN 自定义优化
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 过滤 TensorFlow 日志信息
tf.get_logger().setLevel(logging.ERROR)
def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition Training", add_help=add_help)
    parser.add_argument("--data-path", default=r"E:\BUAA\python+pycharm\FingerNumberTest\Dataset", type=str, help="Path to the dataset")
    parser.add_argument("--batch-size", default=2, type=int, help="Batch size")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cuda or cpu)")
    parser.add_argument("--output-dir", default="./Result", type=str, help="Path to save outputs")
    return parser

# def save_comparison_image(inputs, labels, outputs, save_dir, epoch, class_names):
#     # Convert tensor to numpy for visualization
#     inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (N, H, W, C)
#     labels = labels.cpu().numpy()
#     outputs = outputs.cpu().numpy()
#
#     for i in range(len(inputs)):  # Process each image in the batch
#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#
#         # Original Image
#         axes[0].imshow(inputs[i].squeeze(), cmap='gray')  # Display as grayscale
#         axes[0].set_title(f"Original Image")
#         axes[0].axis('off')
#
#         # Ground Truth
#         axes[1].imshow(labels[i], cmap='gray')  # Assuming labels are integers and can be shown as grayscale
#         axes[1].set_title(f"Ground Truth: {class_names[labels[i]]}")
#         axes[1].axis('off')
#
#         # Prediction
#         predicted_label = np.argmax(outputs[i], axis=0)  # Assuming multi-class classification
#         axes[2].imshow(predicted_label, cmap='gray')
#         axes[2].set_title(f"Prediction: {class_names[predicted_label]}")
#         axes[2].axis('off')
#
#         plt.tight_layout()
#         plt.savefig(f"{save_dir}/comparison_epoch_{epoch}_sample_{i}.png")
#         plt.close()

def main(args):
    device = args.device
    data_dir = args.data_path
    result_dir = args.output_dir

    # Logger
    logger, log_dir = utils.make_logger(result_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_set = fingergraph(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    val_set = fingergraph(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    test_set = fingergraph(data_dir=os.path.join(data_dir, 'test'), transform=transform)

    # DataLoader
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model Setup
    model = GestureRecognitionModel().to(device)

    # Get the optimizer, scheduler, and loss function from the Optimizer script
    criterion = get_criterion()  # 获取损失函数
    optimizer, scheduler = get_optimizer(model, lr=args.lr, scheduler_type="StepLR")  # 获取优化器和调度器

    # Training Loop
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_preds / total_preds

        # Validation
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct_preds / total_preds

        # Logging
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}], "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Accuracy', train_acc, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))

        # Update learning rate
        scheduler.step()

    # Test Evaluation and save comparison images
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    model.eval()

    # Create directory to save comparison images
    save_dir = os.path.join(log_dir, 'comparison_images')
    os.makedirs(save_dir, exist_ok=True)

    test_correct_preds = 0
    test_total_preds = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # # Save comparison images for the first few samples
            # if i < 5:  # Save for the first 5 batches (or adjust as needed)
            #     save_comparison_image(inputs, labels, outputs, save_dir, epoch, class_names=['num1', 'num2', 'num3', 'num4', 'num5'])

            # Accuracy calculation
            _, preds = torch.max(outputs, 1)
            test_correct_preds += torch.sum(preds == labels).item()
            test_total_preds += labels.size(0)

    test_acc = test_correct_preds / test_total_preds
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    writer.close()
    logger.info("Training complete.")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    utils.setup_seed(42)  # Set a random seed
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
    warnings.filterwarnings("ignore", message="oneDNN custom operations are on")

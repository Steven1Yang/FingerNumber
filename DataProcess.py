import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms

class fingergraph(Dataset):
    def __init__(self, data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_info = []
        self.str2int = {"num1": 0, "num2": 1, "num3": 2, "num4": 3, "num5": 4}
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise FileNotFoundError("\n data_dir:{} is a empty dir! Please check your path to images!".format(
                self.data_dir))
        return len(self.img_info)

    def _get_img_info(self):
        for root_dir,folder_dir,file_dir in os.walk(self.data_dir):
            for file in file_dir:
                if file.endswith(".jpg"):
                    path_img = os.path.join(root_dir,file)
                    class_dir = os.path.basename(root_dir)
                    class_num = self.str2int[class_dir]
                    self.img_info.append((path_img,class_num))


if __name__ == "__main__":
    train_root_dir = r"E:\BUAA\python+pycharm\FingerNumberTest\Dataset\train"
    val_root_dir = r"E:\BUAA\python+pycharm\FingerNumberTest\Dataset\val"
    test_root_dir = r"E:\BUAA\python+pycharm\FingerNumberTest\Dataset\test"
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5],[0.5])])
    train_dataset = fingergraph(data_dir=train_root_dir,transform=transform)
    val_dataset = fingergraph(data_dir=val_root_dir,transform=transform)
    test_dataset = fingergraph(data_dir=test_root_dir,transform=transform)

    train_loader = DataLoader(dataset=train_dataset,batch_size=2,pin_memory=True,drop_last=False,shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=2,shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,batch_size=2,shuffle=False)
    # 测试数据加载是否正常
    for i, (inputs, target) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Labels: {target}")

        # 显示一张图片（转换为 numpy 格式）
        img = inputs[0].numpy().transpose(1, 2, 0)  # 转换为 (H, W, C) 格式
        plt.imshow(img.squeeze(), cmap='gray')  # 灰度图
        plt.show()

        if i == 1:  # 打印前两个批次数据后退出循环
            break
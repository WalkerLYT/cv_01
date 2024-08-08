import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义自定义数据集类
class LocationDataset(Dataset):
    def __init__(self, images_folder, annotations_file, transform=None):
        self.images_folder = images_folder  # 图像文件夹路径
        # 读取位置信息文件
        self.annotations = pd.read_csv(annotations_file, delim_whitespace=True, header=None,
                                       names=['image', 'latitude', 'longitude'])
        self.transform = transform  # 数据预处理方法

    def __len__(self):
        return len(self.annotations)  # 返回数据集的长度

    def __getitem__(self, idx):
        # 获取图像文件名
        img_name = os.path.join(self.images_folder, self.annotations.iloc[idx, 0])
        # 打开图像
        image = Image.open(img_name)
        # 获取对应的纬度和经度
        latitude = self.annotations.iloc[idx, 1]
        longitude = self.annotations.iloc[idx, 2]

        # 进行数据预处理
        if self.transform:
            image = self.transform(image)

        # 返回图像和位置信息
        return image, torch.tensor([latitude, longitude], dtype=torch.float32)


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    # 归一化：使用ImageNet的均值和标准差
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载自定义数据集
dataset = LocationDataset(images_folder='../Cambridge', annotations_file='../ImageCollectionCoordinates.txt', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 数据加载器

# 加载预训练的 ResNet-34 模型，并修改输出层
resnet34 = models.resnet34(pretrained=True)
num_ftrs = resnet34.fc.in_features  # 获取全连接层输入特征数
resnet34.fc = nn.Linear(num_ftrs, 2)  # 修改为2个输出节点（纬度和经度）
resnet34.to(device)  # 将模型移到设备（GPU/CPU）

# 设置损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(resnet34.parameters(), lr=0.001)  # 使用Adam优化器

# 训练模型
num_epochs = 25  # 训练轮数
for epoch in range(num_epochs):
    resnet34.train()  # 切换到训练模式
    running_loss = 0.0  # 初始化损失

    for images, locations in dataloader:
        images, locations = images.to(device), locations.to(device)  # 将数据移到设备

        optimizer.zero_grad()  # 清零梯度
        outputs = resnet34(images)  # 前向传播
        loss = criterion(outputs, locations)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累加损失

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")  # 打印损失
    torch.save(resnet34.state_dict(), './models/resnet34_VPR.pth')

print("Finished Training, Model saved to './models/resnet34_VPR.pth'")  # 训练完成
#
# # 切换模型到评估模式
# resnet34.eval()
#
# # 加载验证数据集
# # 假设验证数据集也在 './data/val_images' 和 './data/val_locations.txt'
# val_dataset = LocationDataset(images_folder='./data/val_images', annotations_file='./data/val_locations.txt',
#                               transform=transform)
# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证数据加载器
#
# # 评估模型
# total_loss = 0.0  # 初始化总损失
# with torch.no_grad():  # 禁用梯度计算
#     for images, locations in val_dataloader:
#         images, locations = images.to(device), locations.to(device)  # 将数据移到设备
#         outputs = resnet34(images)  # 前向传播
#         loss = criterion(outputs, locations)  # 计算损失
#         total_loss += loss.item()  # 累加损失
#
# print(f"Validation Loss: {total_loss / len(val_dataloader):.4f}")  # 打印验证损失

import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from PIL import Image




def refacor_data():
    # 定义源目录和目标目录
    source_dirs = ['../round0_train', '../round0_eval']
    train_target_dir = '../data/train'
    val_target_dir = '../data/val'

    # 创建目标目录结构
    for i in range(20):  # 假设类别是 00 到 19
        train_class_dirs = os.path.join(train_target_dir, f"{i:02}")
        val_class_dir = os.path.join(val_target_dir, f"{i:02}")
        os.makedirs(train_class_dirs, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

    # 设置分割比例
    train_ratio = 0.8

    # 按比例将数据分成训练集和验证集
    for source_dir in source_dirs:
        for class_dir in os.listdir(source_dir):
            source_class_dir = os.path.join(source_dir, class_dir)
            train_class_dir = os.path.join(train_target_dir, class_dir)
            val_class_dir = os.path.join(val_target_dir, class_dir)

            if os.path.isdir(source_class_dir):
                files = os.listdir(source_class_dir)
                random.shuffle(files)  # 随机打乱文件

                # 计算训练集和验证集的分界点
                split_idx = int(len(files) * train_ratio)
                train_files = files[:split_idx]
                val_files = files[split_idx:]

                # 将训练文件复制到训练集文件夹
                for file_name in train_files:
                    source_file = os.path.join(source_class_dir, file_name)
                    target_file = os.path.join(train_class_dir, file_name)
                    shutil.copy2(source_file, target_file)

                # 将验证文件复制到验证集文件夹
                for file_name in val_files:
                    source_file = os.path.join(source_class_dir, file_name)
                    target_file = os.path.join(val_class_dir, file_name)
                    shutil.copy2(source_file, target_file)

                print(f"已将 {class_dir} 类别数据分成训练集和验证集，并复制到目标目录")

    print("数据划分完成！")

def train_classification():
    '''加载数据'''
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])

    # 加载训练和验证数据集
    train_dataset = datasets.ImageFolder(root='../data/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='../data/val', transform=transform)

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    '''微调'''
    # 加载预训练的 ResNet18 模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 20)  # 修改全连接层为 20 类

    # 将模型移到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 使用较小的学习率

    # 微调训练代码
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练循环
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 每个 epoch 的训练损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # 验证集上的评估
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    '''保存'''
    torch.save(model.state_dict(), '../model/resnet18_finetuned.pth')


if __name__ == '__main__':
    # refacor_data()
    train_classification()
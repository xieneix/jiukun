import torch
import scipy.io as sio
import os
from torchvision import transforms
from PIL import Image
import shutil
import torchvision.models as models
import numpy as np
from scipy.io import loadmat, savemat
import subprocess
import glob
import random
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed


def collect_all_data():
    # 定义源文件夹和目标文件夹
    train_dir = '../data/train'
    val_dir = '../data/val'
    target_dir = '../data/all'

    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    # 需要复制的类别文件夹范围
    classes_to_copy = [f"{i:02}" for i in range(20)]  # 00-19

    # 定义一个函数来复制文件
    def copy_files(source_dir, target_dir, classes):
        for class_folder in classes:
            source_class_dir = os.path.join(source_dir, class_folder)
            if os.path.isdir(source_class_dir):
                for file_name in os.listdir(source_class_dir):
                    source_file = os.path.join(source_class_dir, file_name)
                    target_file = os.path.join(target_dir, file_name)
                    shutil.copy2(source_file, target_file)
                print(f"已复制 {class_folder} 类别的数据到 {target_dir}")

    # 复制 train 和 val 文件夹中的 00-19 类别文件
    copy_files(train_dir, target_dir, classes_to_copy)
    copy_files(val_dir, target_dir, classes_to_copy)

    print("文件复制完成！")


def extract_and_save_features_client():
    def extract_and_save_feature(image_path, save_path):
        # 加载和预处理图像
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # 添加 batch 维度

        # 提取特征
        with torch.no_grad():
            feature = feature_extractor(image).cpu().numpy()

        # 保存为 .mat 格式
        sio.savemat(save_path, {'fc7': feature})

    # 加载训练好的模型
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 20)  # 假设是 20 类别
    state_dict = torch.load('../model/resnet18_finetuned.pth')  # 确保文件路径正确
    model.load_state_dict(state_dict)  # 将权重加载到模型中
    model.eval()

    # 设置特征提取的层
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # 提取最后的全连接层之前的特征

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 输入图像文件夹路径
    input_dir = '../data/all'  # 图像文件夹路径
    output_dir = '../openmax/data/train_features'  # 保存特征的文件夹路径
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        # 获取类别名（假设文件名格式为 labelXX_imgname.jpg，提取 labelXX 作为类别）
        category = img_name.split('_')[0]
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        # 提取特征并保存到类别文件夹
        img_path = os.path.join(input_dir, img_name)
        save_path = os.path.join(category_dir, f"{os.path.splitext(img_name)[0]}.mat")
        extract_and_save_feature(img_path, save_path)


def extract_and_save_unknown_features(unknown_image_dir='../round0_eval/20', output_dir='../openmax/data/unknown_features'):
    """
    提取未知类图像的特征，并将其保存到指定目录。

    参数:
        unknown_image_dir (str): 未知图像文件夹路径
        output_dir (str): 保存未知类特征的文件夹路径
    """

    # 加载预训练的 ResNet 模型
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 20)  # 假设是 20 类别
    state_dict = torch.load('../model/resnet18_finetuned.pth')  # 确保文件路径正确
    model.load_state_dict(state_dict)  # 将权重加载到模型中
    model.eval()

    # 设置特征提取层
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # 提取最后的全连接层之前的特征

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    os.makedirs(output_dir, exist_ok=True)

    # 遍历未知图像目录
    for img_name in os.listdir(unknown_image_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):  # 确保文件是图像
            img_path = os.path.join(unknown_image_dir, img_name)
            save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.mat")

            # 加载和预处理图像
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)  # 添加 batch 维度

            # 提取特征
            with torch.no_grad():
                feature = feature_extractor(image).cpu().numpy()

            # 保存为 .mat 格式
            sio.savemat(save_path, {'fc7': feature})
            print(f"已保存 {img_name} 的特征到 {save_path}")


def calculate_MAV():
    """
    计算每个类别的平均激活向量 (MAV)，并保存为类别名键名的 .mat 文件（如 'label00'）。
    """
    # 特征文件存放目录
    feature_dir = '../openmax/data/train_features'
    classes = [f"label{i:02}" for i in range(20)]  # 类别名从 label00 到 label19

    # 初始化字典，用于存储每个类别的特征列表
    features = {}

    # 遍历每个类别文件夹，加载该类别的所有特征
    for class_name in classes:
        class_dir = os.path.join(feature_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"正在处理类别: {class_name}")
            features[class_name] = []
            for feature_file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, feature_file)
                try:
                    data = loadmat(file_path)
                    if 'fc7' in data:
                        features[class_name].append(data['fc7'].flatten())  # 将特征展平后添加到类别列表中
                    else:
                        print(f"文件 {file_path} 中未找到 'fc7' 键，跳过。")
                except Exception as e:
                    print(f"加载文件 {file_path} 时出错: {e}")

    # 检查是否所有类别都有特征
    for class_name in classes:
        if class_name not in features or len(features[class_name]) == 0:
            print(f"类别 {class_name} 中没有有效的特征，跳过该类别的 MAV 计算。")
            continue

    # 计算每个类别的平均激活向量 (MAV)
    mav = {}
    for class_name, feature_list in features.items():
        if len(feature_list) > 0:
            mav[class_name] = np.mean(feature_list, axis=0)
            print(f"类别 {class_name} 的 MAV 计算完成，向量维度: {mav[class_name].shape}")
        else:
            print(f"类别 {class_name} 的特征列表为空，无法计算 MAV。")

    # 保存 MAV 到输出目录，键名为类别名
    output_dir = '../openmax/data/mean_files'
    os.makedirs(output_dir, exist_ok=True)
    for class_name, mean_vector in mav.items():
        save_path = os.path.join(output_dir, f"{class_name}.mat")
        try:
            savemat(save_path, {class_name: mean_vector})  # 键名为类别名
            print(f"{class_name} 类的 MAV 已保存到 {save_path}，键名为 '{class_name}'")
        except Exception as e:
            print(f"保存 {class_name} 类的 MAV 时出错: {e}")

    print("所有类别的 MAV 计算和保存完成。")


def calculate_MAV_with_validation_exclusion():
    """
    计算每个类别的平均激活向量 (MAV)，排除验证集样本，保存为类别名键名的 .mat 文件。
    """
    # 特征文件存放目录
    feature_dir = '../openmax/data/train_features'
    val_dir = '../data/val'  # 验证集目录
    classes = [f"label{i:02}" for i in range(20)]  # 类别名从 label00 到 label19

    # 初始化字典，用于存储每个类别的特征列表
    features = {}

    # 收集验证集样本的文件名（不带扩展名）
    val_files = {}
    for class_id, class_name in enumerate(classes):
        val_class_dir = os.path.join(val_dir, f"{class_id:02}")
        if os.path.isdir(val_class_dir):
            val_files[class_name] = set(
                os.path.splitext(file_name)[0]
                for file_name in os.listdir(val_class_dir)
            )
        else:
            val_files[class_name] = set()

    # 遍历每个类别文件夹，加载该类别的所有特征
    for class_name in classes:
        class_dir = os.path.join(feature_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"正在处理类别: {class_name}")
            features[class_name] = []
            for feature_file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, feature_file)
                feature_base_name = os.path.splitext(feature_file)[0]

                # 跳过验证集样本
                if feature_base_name in val_files[class_name]:
                    print(f"跳过验证集样本: {file_path}")
                    continue

                try:
                    data = loadmat(file_path)
                    if 'fc7' in data:
                        features[class_name].append(data['fc7'].flatten())  # 将特征展平后添加到类别列表中
                    else:
                        print(f"文件 {file_path} 中未找到 'fc7' 键，跳过。")
                except Exception as e:
                    print(f"加载文件 {file_path} 时出错: {e}")

    # 检查是否所有类别都有特征
    for class_name in classes:
        if class_name not in features or len(features[class_name]) == 0:
            print(f"类别 {class_name} 中没有有效的特征，跳过该类别的 MAV 计算。")
            continue

    # 计算每个类别的平均激活向量 (MAV)
    mav = {}
    for class_name, feature_list in features.items():
        if len(feature_list) > 0:
            mav[class_name] = np.mean(feature_list, axis=0)
            print(f"类别 {class_name} 的 MAV 计算完成，向量维度: {mav[class_name].shape}")
        else:
            print(f"类别 {class_name} 的特征列表为空，无法计算 MAV。")

    # 保存 MAV 到输出目录，键名为类别名
    output_dir = '../openmax/data/mean_files'
    os.makedirs(output_dir, exist_ok=True)
    for class_name, mean_vector in mav.items():
        save_path = os.path.join(output_dir, f"{class_name}.mat")
        try:
            savemat(save_path, {class_name: mean_vector})  # 键名为类别名
            print(f"{class_name} 类的 MAV 已保存到 {save_path}，键名为 '{class_name}'")
        except Exception as e:
            print(f"保存 {class_name} 类的 MAV 时出错: {e}")

    print("所有类别的 MAV 计算和保存完成。")


def calculate_distance_distribution():
    """
    计算每个类别的特征与其 MAV 的距离分布，并保存到 `openmax/data/mean_distance_files/` 目录中。
    """
    print("正在计算距离分布...")

    # 定义各个路径
    feature_base_dir = os.path.abspath('../openmax/data/train_features')
    mav_base_dir = os.path.abspath('../openmax/data/mean_files')
    output_dir = os.path.abspath('../openmax/data/mean_distance_files')
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有类别
    classes = [f"label{i:02}" for i in range(20)]
    for class_name in classes:
        synset_id = class_name
        mav_file = os.path.join(mav_base_dir, f"{class_name}.mat")
        feature_path = os.path.join(feature_base_dir, class_name)

        # 检查文件和目录是否存在
        if not os.path.exists(mav_file):
            print(f"警告: 找不到 MAV 文件 {mav_file}，跳过该类别")
            continue
        if not os.path.isdir(feature_path):
            print(f"警告: 找不到特征文件目录 {feature_path}，跳过该类别")
            continue

        # 运行 compute_distances.py 脚本，传入参数
        subprocess.run([
            "python", os.path.abspath("../openmax/preprocessing/compute_distances.py"),
            synset_id, mav_file, feature_path
        ], check=True)
        print(f"{class_name} 类的距离分布计算完成")

    print("所有类别的距离分布计算完成并保存到 openmax/data/mean_distance_files/ 目录中")


def calculate_distance_distribution_with_validation_exclusion():
    """
    计算每个类别的特征与其 MAV 的距离分布，排除验证集样本，并保存到 `openmax/data/mean_distance_files/` 目录中。
    """
    print("正在计算距离分布...")

    # 定义各个路径
    feature_base_dir = os.path.abspath('../openmax/data/train_features')
    mav_base_dir = os.path.abspath('../openmax/data/mean_files')
    output_dir = os.path.abspath('../openmax/data/mean_distance_files')
    val_dir = os.path.abspath('../data/val')  # 验证集路径
    os.makedirs(output_dir, exist_ok=True)

    # 收集验证集样本文件名
    val_files = {}
    classes = [f"label{i:02}" for i in range(20)]
    for class_id, class_name in enumerate(classes):
        val_class_dir = os.path.join(val_dir, f"{class_id:02}")
        if os.path.isdir(val_class_dir):
            val_files[class_name] = set(
                os.path.splitext(file_name)[0]
                for file_name in os.listdir(val_class_dir)
            )
        else:
            val_files[class_name] = set()

    # 遍历所有类别
    for class_name in classes:
        synset_id = class_name
        mav_file = os.path.join(mav_base_dir, f"{class_name}.mat")
        feature_path = os.path.join(feature_base_dir, class_name)

        # 检查文件和目录是否存在
        if not os.path.exists(mav_file):
            print(f"警告: 找不到 MAV 文件 {mav_file}，跳过该类别")
            continue
        if not os.path.isdir(feature_path):
            print(f"警告: 找不到特征文件目录 {feature_path}，跳过该类别")
            continue

        # 创建一个临时目录用于存储排除验证集后的特征
        temp_feature_dir = os.path.join(output_dir, f"temp_features_{class_name}")
        os.makedirs(temp_feature_dir, exist_ok=True)

        # 排除验证集样本并复制特征文件到临时目录
        for feature_file in os.listdir(feature_path):
            feature_base_name = os.path.splitext(feature_file)[0]
            if feature_base_name not in val_files[class_name]:  # 排除验证集样本
                src_path = os.path.join(feature_path, feature_file)
                dst_path = os.path.join(temp_feature_dir, feature_file)
                os.link(src_path, dst_path)  # 使用硬链接提高效率

        # 运行 compute_distances.py 脚本，传入参数
        try:
            subprocess.run([
                "python", os.path.abspath("../openmax/preprocessing/compute_distances.py"),
                synset_id, mav_file, temp_feature_dir
            ], check=True)
            print(f"{class_name} 类的距离分布计算完成")
        except Exception as e:
            print(f"{class_name} 类的距离分布计算失败: {e}")

        # 删除临时目录
        for temp_file in os.listdir(temp_feature_dir):
            os.remove(os.path.join(temp_feature_dir, temp_file))
        os.rmdir(temp_feature_dir)

    print("所有类别的距离分布计算完成并保存到 openmax/data/mean_distance_files/ 目录中")


def fit_weibull_and_calculate_openmax(image_feature_path, tail_size, alpha_rank, threshold, output_path=None):
    print(f"正在对 {image_feature_path} 进行 Weibull 拟合并计算 OpenMax 概率...")

    # 定义路径
    mean_files_path = os.path.abspath('../openmax/data/mean_files')
    distance_path = os.path.abspath('../openmax/data/mean_distance_files')

    # 运行 compute_openmax.py 脚本
    result = subprocess.run(
        [
            "python", "../openmax/compute_openmax.py",
            "--image_arrname", image_feature_path,
            "--mean_files_path", mean_files_path,
            "--distance_path", distance_path,
            "--weibull_tailsize", str(tail_size),
            "--alpha_rank", str(alpha_rank)
        ],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    # 保存结果到文件
    if output_path is not None:
        with open(output_path, "a") as f:
            f.write(f"Results for {image_feature_path}:\n")
            f.write(result.stdout)
            f.write("\n" + "-" * 50 + "\n")

    # print("compute_openmax.py 输出内容：")
    # print(result.stdout)

    final_class, final_prob = "unknown", 0.0
    try:
        lines = result.stdout.split("\n")
        fc8_scores = []
        reading_scores = False

        # 遍历输出内容，提取 OpenMax FC8 Scores
        for line in lines:
            if "OpenMax FC8 Scores:" in line:
                reading_scores = True  # 开始读取分数
                continue
            if reading_scores:
                if ("[" in line or "]" in line or line.strip())and 'O' not in line:  # 确保读取非空行或带括号的行
                    # 提取分数行中的数字
                    scores_str = line.strip("[] \n")
                    fc8_scores.extend([float(x) for x in scores_str.split()])
                else:  # 碰到空行或非矩阵格式时，结束读取
                    break

        if fc8_scores:
            # print(f"解析的 OpenMax FC8 Scores: {fc8_scores}")
            max_score = max(fc8_scores)
            max_index = fc8_scores.index(max_score)

            if max_score > threshold:  # 判断是否为未知类
                final_class = f"label{max_index:02}"
                final_prob = max_score/sum(fc8_scores)
            else:
                final_class = 'unknown'
                # print("OpenMax FC8 Scores 总和较小，标记为未知类。")
        else:
            print("Warning: OpenMax FC8 Scores 为空，标记为未知类。")
    except Exception as e:
        print(f"Error parsing OpenMax FC8 Scores: {e}")

    print(f"最终结果验证完成, final class: {final_class}, final prob: {final_prob}")
    return final_class, final_prob, fc8_scores



def validate_unknown_classes(unknown_feature_dir):
    """
    使用 OpenMax 检测未知类图像。

    参数:
        unknown_feature_dir (str): 包含未知类图像特征文件的目录
    """
    print("正在验证未知类图像...")
    for feature_file in os.listdir(unknown_feature_dir):
        if feature_file.endswith(".mat"):
            feature_path = os.path.abspath(os.path.join(unknown_feature_dir, feature_file))
            max_class, max_prob = fit_weibull_and_calculate_openmax(feature_path)
            print(f"未知类验证完成, max classes: {max_class}, max prob: {max_prob}")


def validate_openmax(model_func=fit_weibull_and_calculate_openmax, unknown_features_dir="../openmax/data/unknown_features", val_features_dir="../openmax/data/train_features", val_sample_dir= "../data/val",
                     output_path="validation_results.txt"):
    """
    验证 OpenMax 模型对未知类和验证集已知类的分类性能。

    参数:
        model_func (callable): OpenMax 模型函数（如 `fit_weibull_and_calculate_openmax`）。
        unknown_features_dir (str): 未知类样本特征目录路径。
        val_features_dir (str): 已知类特征目录路径。
        val_sample_dir (str): 用于验证的已知类样本编号目录路径。
        output_path (str): 保存验证结果的文件路径。
    """
    # 初始化统计信息
    total_samples = 0
    correct_predictions = 0

    # 创建结果文件
    with open(output_path, "w") as output_file:
        # 处理未知类样本
        output_file.write("验证未知类样本:\n")
        unknown_files = glob.glob(os.path.join(unknown_features_dir, "*.mat"))
        unknown_correct = 0

        for file_path in unknown_files:
            predicted_class, _, scores = model_func(file_path)
            is_correct = (predicted_class == "unknown")
            unknown_correct += int(is_correct)
            total_samples += 1
            correct_predictions += int(is_correct)

            output_file.write(f"{file_path} -> Predicted: {predicted_class}, Correct: {is_correct}, Scores:{scores}\n")

        output_file.write(f"未知类准确率: {unknown_correct}/{len(unknown_files)}\n\n")

        # 处理验证集已知类样本
        output_file.write("验证已知类样本:\n")
        for class_dir in os.listdir(val_sample_dir):
            val_class_dir = os.path.join(val_sample_dir, class_dir)
            if not os.path.isdir(val_class_dir):
                continue

            feature_class_dir = 'label' + class_dir
            feature_dir = os.path.join(val_features_dir, feature_class_dir)

            # 加载验证集样本编号
            for img_file in os.listdir(val_class_dir):
                sample_name = os.path.splitext(img_file)[0] + ".mat"
                feature_path = os.path.join(feature_dir, sample_name)
                if not os.path.exists(feature_path):
                    continue

                predicted_class, _, scores = model_func(feature_path)
                is_correct = (predicted_class == 'label'+class_dir)
                correct_predictions += int(is_correct)
                total_samples += 1

                output_file.write(f"{feature_path} -> Predicted: {predicted_class}, Correct: {is_correct}, Scores:{scores}\n")

        # 总结统计结果
        overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        output_file.write(
            f"\n总样本: {total_samples}, 正确预测: {correct_predictions}, 总体准确率: {overall_accuracy:.4f}\n")

    print(f"验证完成，结果保存到 {output_path}")


def optimize_and_validate(
    unknown_features_dir="../openmax/data/unknown_features",
    val_features_dir="../openmax/data/train_features",
    val_sample_dir="../data/val",
    weibull_tail_sizes=[60, 100, 200],
    alpha_ranks=[3, 5, 10],
    score_thresholds=[1e-6, 1e-5, 1e-4],
    test_ratio=0.3,
    max_workers=10
):
    """
    优化 OpenMax 参数并在验证集上评估最优参数性能。

    参数：
        unknown_features_dir (str): 未知类样本特征目录路径。
        val_features_dir (str): 已知类特征目录路径。
        val_sample_dir (str): 用于验证的已知类样本编号目录路径。
        weibull_tail_sizes (list): 待测试的 WEIBULL_TAIL_SIZE 参数值。
        alpha_ranks (list): 待测试的 ALPHA_RANK 参数值。
        score_thresholds (list): 待测试的阈值。
        test_ratio (float): 测试集比例。
        max_workers (int): 最大线程数。
    """

    def split_data(file_list, test_ratio=0.3, shuffle=True):
        if shuffle:
            random.shuffle(file_list)
        split_index = int(len(file_list) * (1 - test_ratio))
        return file_list[:split_index], file_list[split_index:]

    # 生成参数组合
    param_grid = list(itertools.product(weibull_tail_sizes, alpha_ranks, score_thresholds))

    # 获取未知类样本文件列表
    unknown_files = glob.glob(os.path.join(unknown_features_dir, "*.mat"))
    unknown_train, unknown_val = split_data(unknown_files, test_ratio)

    # 获取已知类样本文件列表
    known_samples = []
    for class_dir in os.listdir(val_sample_dir):
        val_class_dir = os.path.join(val_sample_dir, class_dir)
        if not os.path.isdir(val_class_dir):
            continue
        feature_class_dir = 'label' + class_dir
        feature_dir = os.path.join(val_features_dir, feature_class_dir)
        for img_file in os.listdir(val_class_dir):
            sample_name = os.path.splitext(img_file)[0] + ".mat"
            feature_path = os.path.join(feature_dir, sample_name)
            if os.path.exists(feature_path):
                known_samples.append(feature_path)

    known_train, known_val = split_data(known_samples, test_ratio)

    # 定义单个参数组合的评估函数
    def evaluate_params(weibull_tail_size, alpha_rank, score_threshold):
        correct_predictions, total_samples = 0, 0
        for file_path in unknown_train:
            predicted_class, _, _ = fit_weibull_and_calculate_openmax(
                file_path, weibull_tail_size, alpha_rank, score_threshold)
            correct_predictions += int(predicted_class == "unknown")
            total_samples += 1
        for file_path in known_train:
            predicted_class, _, _ = fit_weibull_and_calculate_openmax(
                file_path, weibull_tail_size, alpha_rank, score_threshold)
            true_class = file_path.split("\\")[-1].split('_')[0]
            print(true_class, predicted_class)
            correct_predictions += int(predicted_class == true_class)
            total_samples += 1
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        return (weibull_tail_size, alpha_rank, score_threshold, accuracy)

    # 多线程处理参数组合
    best_params, best_accuracy = None, 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(evaluate_params, w, a, t): (w, a, t)
            for w, a, t in param_grid
        }
        for future in as_completed(future_to_params):
            weibull_tail_size, alpha_rank, score_threshold, accuracy = future.result()
            print(f"参数组合: WEIBULL_TAIL_SIZE={weibull_tail_size}, ALPHA_RANK={alpha_rank}, 阈值={score_threshold}, 准确率={accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (weibull_tail_size, alpha_rank, score_threshold)

    print(f"\n最优参数: WEIBULL_TAIL_SIZE={best_params[0]}, ALPHA_RANK={best_params[1]}, 阈值={best_params[2]}")
    print(f"最高准确率: {best_accuracy:.4f}\n")

    # 在验证集上评估最优参数
    WEIBULL_TAIL_SIZE, ALPHA_RANK, THRESHOLD = best_params[0], best_params[1], best_params[2]
    correct_predictions, total_samples = 0, 0
    for file_path in unknown_val:
        predicted_class, _, _ = fit_weibull_and_calculate_openmax(
            file_path, WEIBULL_TAIL_SIZE, ALPHA_RANK, THRESHOLD)
        correct_predictions += int(predicted_class == "unknown")
        total_samples += 1
    for file_path in known_val:
        predicted_class, _, _ = fit_weibull_and_calculate_openmax(
            file_path, WEIBULL_TAIL_SIZE, ALPHA_RANK, THRESHOLD)
        true_class = file_path.split("\\")[-1].split('_')[0]
        correct_predictions += int(predicted_class == true_class)
        total_samples += 1
    val_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"验证集准确率: {val_accuracy:.4f}")


def load_model(model_path='../model/resnet18_finetuned.pth'):
    """
    加载 ResNet18 模型并返回模型及特征提取器。
    """
    # 加载训练好的模型
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 20)  # 假设是 20 类别
    state_dict = torch.load(model_path)  # 确保文件路径正确
    model.load_state_dict(state_dict)  # 将权重加载到模型中
    model.eval()

    # 设置特征提取的层
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # 提取最后的全连接层之前的特征

    return model, feature_extractor


def classify(patch, patch_num, feature_extractor):
    """
    对单个图片块进行分类。
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 转换图片块
    patch_tensor = transform(patch).unsqueeze(0)

    # 提取特征
    with torch.no_grad():
        feature = feature_extractor(patch_tensor).cpu().numpy()

    # 保存为 .mat 格式
    feature_output_dir = 'openmax/data/tmp'
    os.makedirs(feature_output_dir, exist_ok=True)
    feature_save_path = os.path.join(feature_output_dir, f"patch{patch_num}.mat")
    sio.savemat(feature_save_path, {'fc7': feature})

    # OpenMax 后处理

    # 获取当前文件所在目录的绝对路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    mean_files_path = os.path.abspath(os.path.join(current_file_dir, '../openmax/data/mean_files'))
    distance_path = os.path.abspath(os.path.join(current_file_dir, '../openmax/data/mean_distance_files'))
    WEIBULL_TAIL_SIZE = 60
    ALPHA_RANK = 10
    THRESHOLD = 1e-6



    # 构造 compute_openmax.py 的绝对路径
    compute_openmax_path = os.path.join(current_file_dir, "../openmax/compute_openmax.py")

    result = subprocess.run(
        [
            "python", compute_openmax_path,
            "--image_arrname", feature_save_path,
            "--mean_files_path", mean_files_path,
            "--distance_path", distance_path,
            "--weibull_tailsize", str(WEIBULL_TAIL_SIZE),
            "--alpha_rank", str(ALPHA_RANK)
        ],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    final_class, final_prob = "unknown", 0.0
    try:
        lines = result.stdout.split("\n")
        fc8_scores = []
        reading_scores = False

        # 提取 OpenMax FC8 Scores
        for line in lines:
            if "OpenMax FC8 Scores:" in line:
                reading_scores = True
                continue
            if reading_scores:
                if ("[" in line or "]" in line or line.strip()) and 'O' not in line:
                    scores_str = line.strip("[] \n")
                    fc8_scores.extend([float(x) for x in scores_str.split()])
                else:
                    break

        if fc8_scores:
            max_score = max(fc8_scores)
            max_index = fc8_scores.index(max_score)

            if max_score > THRESHOLD:
                final_class = max_index
            else:
                final_class = 20
        else:
            print("Warning: OpenMax FC8 Scores 为空，标记为未知类。")
    except Exception as e:
        print(f"Error parsing OpenMax FC8 Scores: {e}")

    return final_class


if __name__ == '__main__':
    '''最优参数: WEIBULL_TAIL_SIZE=60, ALPHA_RANK=10, 阈值=1e-06
    最高准确率: 0.9598
    验证集准确率: 0.9563'''

    # collect_all_data()
    # extract_and_save_features_client()
    # calculate_MAV()
    # calculate_MAV_with_validation_exclusion()
    # calculate_distance_distribution()
    # calculate_distance_distribution_with_validation_exclusion()
    # fit_weibull_and_calculate_openmax()
    # extract_and_save_unknown_features()
    # validate_unknown_classes('../openmax/data/unknown_features')
    # final_class, final_prob = fit_weibull_and_calculate_openmax('../openmax/data/train_features/label01/label01_002.mat')
    # final_class, final_prob = fit_weibull_and_calculate_openmax('../openmax/data/unknown_features/open_1_8_1.mat')
    # print(final_class, final_prob)
    # validate_openmax()
    optimize_and_validate()







import os
import shutil
import random

# 设置路径
dataset_path = 'C:/Users/hongc/Desktop/for vs code/dataset'
images_dir = os.path.join(dataset_path, 'images')
labels_dir = os.path.join(dataset_path, 'labels')

# 1. 自动清理旧的 train 和 valid 文件夹，确保重新分配时的纯净性
for folder in ['train', 'valid']:
    target_path = os.path.join(dataset_path, folder)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'labels'), exist_ok=True)

# 2. 获取所有有效图片
files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(files)

# 3. 统计并报错提示：检查图片与标签是否一一对应
img_names = {os.path.splitext(f)[0] for f in files}
lbl_names = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

missing = img_names - lbl_names
if missing:
    print(f"警告：以下 {len(missing)} 张图片缺少对应的标签文件！")
    print(f"缺失列表: {list(missing)}")
    # 你可以选择在这里使用 input() 等待用户确认，或者直接继续
    # input("按回车键继续（部分图片将无法参与训练）...")

# 4. 按 80% 训练, 20% 验证分配
split_idx = int(0.8 * len(files))
train_files = files[:split_idx]
val_files = files[split_idx:]

def copy_files(file_list, split_type):
    for f in file_list:
        # 复制图片
        shutil.copy(os.path.join(images_dir, f), os.path.join(dataset_path, split_type, 'images', f))
        
        # 复制对应的标签
        label_file = os.path.splitext(f)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(dataset_path, split_type, 'labels', label_file))

copy_files(train_files, 'train')
copy_files(val_files, 'valid')

print(f"数据集整理完成！")
print(f"训练集: {len(train_files)} 组")
print(f"验证集: {len(val_files)} 组")
print("请确认 dataset/train 和 dataset/valid 文件夹内容是否正确。")
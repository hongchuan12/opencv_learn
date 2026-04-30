import os
import shutil
import random


dataset_path = 'C:/Users/hongc/Desktop/for vs code/dataset'
images_dir = os.path.join(dataset_path, 'images')
labels_dir = os.path.join(dataset_path, 'labels')


for folder in ['train', 'valid']:
    target_path = os.path.join(dataset_path, folder)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'labels'), exist_ok=True)


files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(files)


img_names = {os.path.splitext(f)[0] for f in files}
lbl_names = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

missing = img_names - lbl_names
if missing:
    print(f"warn： {len(missing)} image miss！")
    print(f"missing list: {list(missing)}")


split_idx = int(0.8 * len(files))
train_files = files[:split_idx]
val_files = files[split_idx:]

def copy_files(file_list, split_type):
    for f in file_list:
     
        shutil.copy(os.path.join(images_dir, f), os.path.join(dataset_path, split_type, 'images', f))
        
        label_file = os.path.splitext(f)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(dataset_path, split_type, 'labels', label_file))

copy_files(train_files, 'train')
copy_files(val_files, 'valid')

print(f"dataset have been sliced！")
print(f"trainset: {len(train_files)} 组")
print(f"validation set: {len(val_files)} 组")

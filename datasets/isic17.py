import os

def rename_files_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否符合格式
        if "_segmentation" in filename:
            # 去除 "_segmentation" 并生成新的文件名
            new_name = filename.replace("_segmentation", "")
            # 获取完整路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            # 重命名文件
            try:
                os.rename(old_path, new_path)
                print(f"已重命名: {filename} -> {new_name}")
            except Exception as e:
                print(f"重命名 {filename} 时出错: {e}")
    print("所有符合条件的文件已重命名。")

# 使用示例
folder_path = "/data1/users/caojiarui/cjr/LB-UNet/data/isic2017/val/masks"  # 替换为你的文件夹路径
rename_files_in_folder(folder_path)

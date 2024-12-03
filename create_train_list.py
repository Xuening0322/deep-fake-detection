import os
import glob
import argparse
from tqdm import tqdm

def create_train_list(root_dir, output_file):
    """
    遍历LRS2数据集目录，创建训练数据列表文件 (使用相对路径)
    Args:
        root_dir: LRS2数据集根目录
        output_file: 输出的列表文件路径
    """
    print(f"Scanning directory: {root_dir}")
    
    # 查找所有 mp4 文件
    video_paths = []
    for ext in ['*.mp4']:
        paths = glob.glob(os.path.join(root_dir, '**', ext), recursive=True)
        print(f"Found {len(paths)} {ext} files")
        video_paths.extend(paths)
    
    # 转换为相对路径
    video_paths = [os.path.relpath(p, start=root_dir) for p in video_paths]
    
    print(f"Total found: {len(video_paths)} videos")
    
    # 写入文件
    with open(output_file, 'w') as f:
        for path in tqdm(video_paths, desc="Writing paths"):
            f.write(f"{path}\n")
    
    print(f"Train list saved to: {output_file}")
    print(f"First few entries:")
    os.system(f"head -n 5 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                      help='LRS2数据集根目录路径')
    parser.add_argument('--output', type=str, default='train_list.txt',
                      help='输出的训练列表文件路径')
    args = parser.parse_args()
    
    create_train_list(args.root_dir, args.output)

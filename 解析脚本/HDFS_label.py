import os
import csv
import re
from settings import parse_settings  # 假设 settings.py 在同一目录下
from tqdm import tqdm  # 导入 tqdm 用于显示进度条


def add_labels(input_file, output_file, label_file, log_type='HDFS_2k'):
    """
    为指定 log_type 的日志文件添加 Label 列。
    
    参数:
    log_type (str): 日志类型。
    """

    # 读取标签文件并构建 BlockId 到 Label 的映射字典
    block_id_label_map = {}
    print("标签数据读取中...")
    with open(label_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_id = row['BlockId']
            label = row['Label']
            block_id_label_map[block_id] = label

    # 按 BlockId 长度降序排序，避免短的 BlockId 错误匹配长的 BlockId
    sorted_block_ids = sorted(block_id_label_map.keys(), key=lambda x: len(x), reverse=True)

    # 编译一个包含所有 BlockId 的正则表达式模式，使用单词边界确保准确匹配
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_block_ids)) + r')\b')

    # 获取文件行数以初始化 tqdm 进度条
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # 减去表头行
        
    # 处理原始日志文件，逐行读取和写入
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        # 检查 parse_settings 是否有指定表头，如果有则使用，否则使用文件默认表头
        if log_type in parse_settings and 'headers' in parse_settings[log_type]:
            headers = parse_settings[log_type]['headers']
        else:
            headers = reader.fieldnames
        
        # 在表头中插入 'Label' 列，位置在 'LineId' 之后
        try:
            line_id_index = headers.index('LineId')
            new_headers = headers[:line_id_index + 1] + ['Label'] + headers[line_id_index + 1:]
        except ValueError:
            # 如果没有 'LineId'，将 'Label' 追加到表头末尾
            new_headers = headers + ['Label']

        writer = csv.DictWriter(outfile, fieldnames=new_headers)
        writer.writeheader()  # 写入表头

        # 逐行处理日志内容
        for row in tqdm(reader, total=total_lines, desc="处理日志文件"):
            content = row['Content']  # 获取日志内容
            label = 'Normal'  # 默认 Label 值
            
            # 使用正则表达式在内容中查找 BlockId
            match = pattern.search(content)
            if match: label = 'Anomaly'  # 实在太忙故用次方法写死变量
                # block_id = match.group(1)  # 获取匹配到的第一个 BlockId
                # label = block_id_label_map[block_id]  # 根据 BlockId 获取对应的 Label

            # 构建新行并插入 Label
            new_row = {field: row.get(field, '') for field in headers}  # 复制原始行数据
            new_row['Label'] = label  # 添加 Label 列
            writer.writerow(new_row)  # 写入新行


if __name__ == "__main__":

    log_type = 'HDFS'
    log_data_dir = './dataset'  # 日志文件所在的目录
    input_dir = os.path.join(log_data_dir, log_type)  # 输入目录
    output_dir = input_dir  # 输出目录与输入目录相同

    input_file = os.path.join(input_dir, f"{log_type}.log_structured.csv")  # 输入日志文件路径
    label_file = os.path.join(input_dir, 'anomaly_blocks.csv')  # 标签文件路径

    raw_log_file = os.path.join(output_dir, f"{log_type}.log_structured_label.csv")  # 输出带 Label 的文件路径

    add_labels(input_file, raw_log_file, label_file, log_type)
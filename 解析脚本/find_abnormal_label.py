import csv

# 输入文件和输出文件路径
input_file = './dataset/HDFS/anomaly_label.csv'
output_file = './dataset/HDFS/anomaly_blocks.csv'

# 打开输入文件进行读取，打开输出文件进行写入
with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    
    # 写入输出文件的标题行
    writer.writeheader()
    
    # 遍历输入文件的每一行
    for row in reader:
        if row['Label'] == 'Anomaly':
            writer.writerow(row)

print(f"筛选完成，结果已保存到 {output_file}")
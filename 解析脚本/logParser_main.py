import csv
import re, string
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from settings import parse_settings  # 配置文件，包含不同日志类型的解析设置
from datetime import datetime
from tqdm import tqdm 
import logging

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log_parser.log"),
        logging.StreamHandler()
    ]
)

# 预定义的时间相关缩写集合（月份、星期、时区）
date_alias = {'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
              'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN',
              'PDT', 'UTC'}

# 默认运行配置参数
default_running_settings = {
    'enable_time_parse': True,       # 是否启用时间解析
    'enable_specific_key': True,     # 是否启用特定键提取
    'enable_regex_substitute': True, # 是否启用正则替换
    'enable_regex_split': True,      # 是否启用正则分割
    'parse_result_output': True,     # 是否输出解析结果
    'parse_detail_output': False,    # 是否输出详细解析过程
    'exclude_dataset': {'Total'},    # 需要排除的数据集
    'specific_dataset': {'BGL'},          # 指定需要处理的数据集（空表示全部）
    'origin_data_type': 'log2csv',       # 原始数据类型（csv/raw/log2csv）
    'log_data_dir': './dataset'         # 日志数据目录
}

class LogParser:
    def __init__(self, settings: Dict[str, Any], 
                 enable_time_parse: bool = True,
                 enable_specific_key: bool = True,
                 enable_regex_substitute: bool = True,
                 enable_regex_split: bool = True):
        self.settings = settings
        self.enable_time_parse = enable_time_parse
        self.enable_specific_key = enable_specific_key
        self.enable_regex_substitute = enable_regex_substitute
        self.enable_regex_split = enable_regex_split
        
        # 状态维护
        self.key_id_map: Dict[Tuple, int] = {}
        self.templates: Dict[int, str] = {}
        self.cluster = defaultdict(list)

        # 进度显示与调试
        self._processed_lines = 0
        self._cluster_stats = defaultdict(int)
    
    def is_variable(self, token: str) -> bool:
        """判断给定token是否为变量/动态值"""
        # 已被标记为变量的情况 # 单个标点符号 # 包含数字 # 十六进制格式 # 时间相关缩写
        return token == '' or \
            token[0] == '<' or \
            (len(token) == 1 and token[0] in string.punctuation) or \
            any(char.isdigit() for char in token) or \
            all('a' <= c.lower() <= 'f' or c.isdigit() for c in token) or \
            token.upper() in date_alias

    def parse_time(self, logline: str, time_regex: str, time_format: str):
        """从日志行中提取并标准化时间信息"""
        match = re.search(time_regex, logline)
        if match:
            data_str = match.group(1)
            # 处理UNIX时间戳格式
            if time_format == '%UNIX_TIMESTAMP':
                date_obj = datetime.utcfromtimestamp(int(data_str))
            else:
                date_obj = datetime.strptime(match.group(1), time_format)
            # 替换时间部分为<DATETIME>标记
            logline = re.sub(time_regex, '<DATETIME>', logline)
            return date_obj, logline
        else:
            return None, logline

    def generate_log_template(self, tokens: List[str]) -> str:
        """生成日志模板，将变量部分替换为<*>，并合并连续的变量"""
        template = []
        prev_is_variable = False
        for token in tokens:
            current_is_variable = self.is_variable(token)
            if current_is_variable:
                if not prev_is_variable:
                    template.append('<*>')
                prev_is_variable = True
            else:
                template.append(token)
                prev_is_variable = False
        return ' '.join(template)

    def save_templates(self, output_path: str) -> None:
        """
        将日志模板保存到指定CSV文件
        格式：
        EventId,EventTemplate
        E0,"template1"
        E1,"template2"
        """
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['EventId', 'EventTemplate'])
            
            # 按事件ID升序排列保证输出顺序
            sorted_templates = sorted(
                self.templates.items(),
                key=lambda x: x[0]
            )
            
            for event_id, template in sorted_templates:
                writer.writerow([
                    f'E{event_id}',  # 生成标准事件ID格式
                    f'{template}'  # 保留原始模板的引号包裹
                ])

    def parse_line(self, raw_line: str) -> Tuple[str, str]:
        """单行日志解析核心逻辑"""
        self._processed_lines += 1
        if self._processed_lines % 1000000 == 0:
            logging.info(f"已处理 {self._processed_lines} 行日志...")

        line = raw_line.strip()
        specific_keys = []

        # 时间解析
        if self.enable_time_parse:
            _, line = self.parse_time(line, self.settings['time_regex'], self.settings['time_format'])

        # 特定键提取
        if self.enable_specific_key:
            for pattern in self.settings['specific']:
                specific_keys.extend(re.findall(pattern, line))

        # 正则替换
        if self.enable_regex_substitute:
            for i, (rex, repl) in enumerate(self.settings['substitute_regex'].items()):
                if i in self.settings.get('replace_once', []):
                    line = re.sub(rex, repl, line, count=1)
                else:
                    line = re.sub(rex, repl, line)

        # 分割处理
        if self.enable_regex_split:
            for rex, repl in self.settings['split_regex'].items():
                line = re.sub(rex, repl, line)

        # 生成tokens和日志键
        tokens = [t for t in line.split() if t]
        static_tokens = [t for t in tokens if not self.is_variable(t)]
        log_key = tuple(static_tokens + specific_keys)

        # 分配事件ID并生成模板
        if log_key not in self.key_id_map:
            event_id = len(self.key_id_map)
            self.key_id_map[log_key] = event_id
            self.templates[event_id] = self.generate_log_template(tokens)
        
        event_id = self.key_id_map[log_key]
        self._cluster_stats[event_id] += 1

        return f"E{event_id}", self.templates[event_id]


def transform_log_to_csv(input_path: str, 
                        output_path: str, 
                        parser: LogParser,
                        chunk_size: int = 5000) -> None:
    """改进后的日志转换函数，支持分块处理和实时解析"""
    # 预读文件获取总行数（用于进度条）
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # 用户自定义列表头部种类
    headers = parser.settings['headers']

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        
        writer = csv.writer(f_out)
        writer.writerow(headers)
        
        chunk = []
        logging.info(f"开始处理文件: {input_path}")

        content_index = len(headers) - 3  # 倒数第三列为Content

        # 添加进度条
        with tqdm(total=total_lines, desc="Processing logs", unit="line") as pbar:
            for line_id, raw_line in enumerate(f_in, 1):
                # 分块处理逻辑
                if line_id % chunk_size == 0:
                    writer.writerows(chunk)
                    chunk = []
                    logging.debug(f"已写入 {line_id} 行到临时文件")

                parts = raw_line.strip().split()

                # 动态验证字段数（保留最后三个字段给Content+EventId+EventTemplate）
                if len(parts) < (content_index - 1):  # LineId不占用parts位置
                    continue

                try:
                    # 动态构建基础结构
                    structured = [line_id]  # LineId
                    # 填充中间字段（从parts按顺序取）
                    for i in range(content_index - 1):  # 跳过LineId和最后三个字段
                        if i < len(parts):
                            structured.append(parts[i])
                        else:
                            structured.append("")  # 补空值
                    
                    # 处理Content（合并剩余部分）
                    content = ' '.join(parts[content_index-1:])  # 注意索引偏移
                    structured.append(content)
                    
                    # 实时解析并填充最后两列
                    event_id, template = parser.parse_line(structured[-1])
                    structured.extend([event_id, template])
                    
                    chunk.append(structured)
                except Exception as e:
                    logging.error(f"处理行 {line_id} 失败: {str(e)}")

                pbar.update(1)  # 更新进度条
        
        # 写入剩余数据
        if chunk:
            writer.writerows(chunk)
        
        logging.info(f"完成处理，共处理 {parser._processed_lines} 行日志")
        logging.info(f"生成 {len(parser.templates)} 个日志模板")
        logging.info(f"事件分布统计: {dict(parser._cluster_stats)}")


def run_benchmark(running_settings: Dict[str, Any]) -> None:
    """改进后的基准测试函数"""
    log_data_dir = Path(running_settings["log_data_dir"])
    
    # 获取所有子目录名称作为数据集类型（过滤非目录项）
    log_dataset_types = [t.name for t in log_data_dir.glob("*") if t.is_dir()]
    
    benchmark_result = []  # 存储各数据集的评估结果

    # 遍历处理每个日志数据集
    for log_type in log_dataset_types:
        # 数据集过滤逻辑
        if log_type in running_settings['exclude_dataset']: 
            continue  # 跳过排除的数据集
        if (running_settings['specific_dataset'] and 
            log_type not in running_settings['specific_dataset']):
            continue  # 跳过不在指定列表中的数据集
        
        # 打印当前处理的数据集名称（对齐格式）
        if running_settings['parse_result_output']: 
            print(f"{log_type:12}", end='')


        # log_type.name = 'BGL2'
        # 初始化日志解析器
        parser = LogParser(
            settings=parse_settings[log_type],
            enable_time_parse=running_settings['enable_time_parse'],
            enable_specific_key=running_settings['enable_specific_key'],
            enable_regex_substitute=running_settings['enable_regex_substitute'],
            enable_regex_split=running_settings['enable_regex_split']
        )
        
        # 构建文件路径
        raw_log_path = log_data_dir / log_type / f"{log_type}.log"  # 原始日志文件
        structured_log_path = log_data_dir / log_type / f"{log_type}.log_structured.csv"  # 结构化日志文件
        output_template_path = log_data_dir / log_type / f"{log_type}.log_template.csv"  # 原始日志文件
           
        # 执行转换和解析
        transform_log_to_csv(
            input_path=str(raw_log_path),
            output_path=str(structured_log_path),
            parser=parser,
            chunk_size=20000  # 20万条数据分10块处理
        )
        
        # 保存模板信息
        parser.save_templates(output_template_path)

if __name__ == "__main__":
    print("begin")
    run_benchmark(default_running_settings)
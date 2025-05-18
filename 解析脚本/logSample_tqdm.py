import pandas as pd
import os
import logging
from tqdm import tqdm
from collections import deque
from settings import parse_settings
from heapq import heappush, heappop

def setup_logging(output_dir, log_type, radio):
    """配置增强型日志记录"""
    log_file = os.path.join(output_dir, f"{log_type}_opt_{radio}_sample.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


import heapq

def dynamic_allocation(distribution, target):
    """
    完全分配目标采样数的改进算法
    Args:
        distribution: 事件分布字典 {event_id: count}
        target: 目标采样总数
    
    Returns:
        allocation: 完全分配后的方案字典
    """
    total_events = sum(distribution.values())
    num_types = len(distribution)
    
    # 输入验证
    if target > total_events:
        raise ValueError(f"目标采样数{target}超过总事件数{total_events}")
    if target < num_types:
        raise ValueError(f"目标采样数{target}小于事件类型数{num_types}")

    # 初始化分配（每个类型至少1个）
    allocation = {e:1 for e in distribution}
    remaining = target - num_types
    
    # 创建最大堆（使用负数实现）
    heap = []
    for event, count in distribution.items():
        available = count - 1  # 已分配1个
        if available > 0:
            heapq.heappush(heap, (-available, event))

    # 执行剩余分配
    while remaining > 0 and heap:
        neg_avail, event = heapq.heappop(heap)
        avail = -neg_avail
        
        # 分配1个样本
        allocation[event] += 1
        remaining -= 1
        avail -= 1
        
        # 重新入堆（如果还有可用量）
        if avail > 0:
            heapq.heappush(heap, (-avail, event))

    if remaining > 0:
        raise RuntimeError(f"分配未完成，剩余{remaining}个未分配")

    return allocation

def proportional_allocation(distribution, target):
    """
    按比例分配且保证完全分配的算法
    Args:
        distribution: 事件分布字典 {event_id: count}
        target: 目标采样总数
    
    Returns:
        allocation: 分配方案字典
    """
    total_events = sum(distribution.values())
    num_types = len(distribution)
    
    # 输入验证
    if target > total_events:
        raise ValueError(f"目标采样数{target}超过总事件数{total_events}")
    if target < num_types:
        raise ValueError(f"目标采样数{target}小于事件类型数{num_types}")

    # 计算每个事件的理论分配比例
    proportions = {e: count/total_events for e, count in distribution.items()}
    
    # 第一阶段：每个事件至少分配1个
    allocation = {e:1 for e in distribution}
    remaining = target - num_types
    
    # 第二阶段：按比例分配剩余数量
    additional_allocation = {}
    for e in distribution:
        expected = proportions[e] * remaining
        additional_allocation[e] = expected
    
    # 处理整数分配
    allocated = 0
    sorted_events = sorted(additional_allocation.items(), 
                         key=lambda x: x[1] - int(x[1]), 
                         reverse=True)  # 按小数部分排序
    
    # 先分配整数部分
    for e, amount in sorted_events:
        integer_part = int(amount)
        allocation[e] += integer_part
        allocated += integer_part
    
    # 分配余数（按小数部分从大到小）
    remainder = remaining - allocated
    for e, amount in sorted_events[:remainder]:
        allocation[e] += 1
    
    # 第三阶段：修正超出实际可用量的情况
    heap = []
    for e in allocation:
        available = distribution[e]
        if allocation[e] > available:
            over = allocation[e] - available
            allocation[e] = available
            heapq.heappush(heap, (-over, e))  # 最大堆存储需要补足的数量
    
    # 重新分配溢出的配额
    while heap:
        need, e = heapq.heappop(heap)
        need = -need
        
        # 寻找可以增加配额的事件
        candidates = []
        for event in distribution:
            if allocation[event] < distribution[event]:
                candidates.append(event)
        
        if not candidates:
            raise RuntimeError("无法完成分配：无可用候选事件")
        
        # 按比例分配需要补足的数量
        candidate_weights = [distribution[e] - allocation[e] for e in candidates]
        total_weight = sum(candidate_weights)
        
        if total_weight == 0:
            raise RuntimeError("无法完成分配：所有事件配额已用尽")
        
        for event in candidates:
            can_add = min(
                need * (candidate_weights[candidates.index(event)] / total_weight),
                distribution[event] - allocation[event]
            )
            allocation[event] += int(can_add)
            need -= int(can_add)
            if need <= 0:
                break
    
    return allocation


def optimized_sampling(log_type, radio, scheme='anomaly_based'):
    """优化后的采样主函数"""
    # 初始化配置
    config = parse_settings[log_type]
    headers = config['headers']
    line_id_col = next(h for h in headers if 'LineId' in h)
    event_id_col = next(h for h in headers if 'EventId' in h)
    
    # 路径配置
    log_data_dir = './dataset'
    input_dir = os.path.join(log_data_dir, log_type)
    raw_log_file = os.path.join(input_dir, f"{log_type}.log_structured.csv")
    output_file = os.path.join(input_dir, f"{log_type}_opt_{radio}.csv")
    
    setup_logging(input_dir, log_type, radio)
    
    # 高效数据读取
    logging.info("🔄 开始读取数据文件...")
    chunks = []
    for chunk in tqdm(pd.read_csv(raw_log_file, chunksize=100000, header=0, names=headers),
                      desc="数据读取进度",
                      unit="chunk"):
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0)
    
    # 数据预处理
    logging.info("🔍 进行数据预处理...")
    if scheme == 'anomaly_based':
        anomaly_condition = df[headers[1]].isin(config['anomaly_labels'])
    else:
        anomaly_condition = ~df[headers[1]].isin(config['normal_labels'])
    
    anomaly_df = df[anomaly_condition].copy()
    normal_df = df[~anomaly_condition].copy()
    total_normal = len(normal_df)
    total_anomaly = len(anomaly_df)
    
    # 计算目标采样数
    original_target_anomaly = total_normal // radio
    target_anomaly = original_target_anomaly
    allocation = None
    logging.info(f"📊 样本平衡参数：正常:异常 = {radio}:1")
    logging.info(f"📈 目标异常样本数：{target_anomaly}（基于{total_normal}正常样本）")
     
    # 异常事件分布分析
    anomaly_dist = anomaly_df[event_id_col].value_counts().to_dict()
    logging.info(f"\n原始异常事件分布（共{len(anomaly_dist)}类）：")
    # for event, count in sorted(anomaly_dist.items(), key=lambda x: x[1], reverse=True):
    #     logging.info(f"  - {event}: {count} 条")
    logging.info(f"{anomaly_dist}")

    # 异常样本不足处理逻辑
    if total_anomaly < original_target_anomaly:
        logging.warning(f"异常样本不足：实际{total_anomaly} < 目标{original_target_anomaly}")
        logging.info("启动应急处理方案：使用全部异常样本，调整正常样本比例")
        target_anomaly = total_anomaly
    else:
        # 生成正常分配方案
        try:
            # 动态分配方案生成
            logging.info("\n生成动态分配方案...")
            allocation = proportional_allocation(anomaly_dist, target_anomaly)
        except (ValueError, RuntimeError) as e:
            logging.error(f"分配失败: {str(e)}")
            return


    # 方案展示和确认
    if allocation is not None:
        actual_total = sum(allocation.values())
        logging.info(f"\n分配方案概览（目标：{target_anomaly}，实际：{actual_total}）:")

        if len(allocation) > 10:
            logging.info("分配方案摘要：")
            logging.info(f"事件类型数：{len(allocation)}")
            logging.info(f"最大单事件采样数：{max(allocation.values())}")
            logging.info(f"最小单事件采样数：{min(allocation.values())}")
            logging.info("完整分配方案：")
            logging.info(allocation)
        else:
            for event, count in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
                original = anomaly_dist[event]
                ratio = count/original if original >0 else 0
                logging.info(f"{event}: {count}/{original} ({ratio:.1%})")
        
        user_input = input("\n确认执行采样？（Y/N） ").strip().upper()
        if user_input != 'Y':
            logging.warning("操作已取消")
            return
    
    # 执行异常样本采样
    logging.info("\n执行异常样本采样...")
    if allocation:
        sampled_anomaly = []
        for event_id, quota in allocation.items():
            event_data = anomaly_df[anomaly_df[event_id_col] == event_id]
            sample_size = min(quota, len(event_data))
            sampled = event_data.sample(n=sample_size, random_state=42)
            sampled_anomaly.append(sampled)
            logging.info(f"{event_id}: 采样 {len(sampled)}/{quota} 条")
        anomaly_sample = pd.concat(sampled_anomaly)
    else:
        anomaly_sample = anomaly_df.copy()
        logging.info(f"使用全部异常样本：{len(anomaly_sample)} 条")

    final_anomaly_count = len(anomaly_sample)
    
    # 计算所需正常样本
    required_normal = final_anomaly_count * radio
    logging.info(f"\n最终需采样正常样本：{required_normal} 条")

    # 正常样本处理
    if len(normal_df) < required_normal:
        logging.error("正常样本不足，无法完成采样")
        raise ValueError(f"正常样本不足：需要{required_normal} 现有{len(normal_df)}")
    
    if len(normal_df) > required_normal:
        drop_count = len(normal_df) - required_normal
        logging.info(f"删除多余正常样本：{drop_count} 条")
        normal_sample = normal_df.sample(n=required_normal, random_state=42)
    else:
        normal_sample = normal_df.copy()
        logging.info("使用全部正常样本")
    
    # 结果整合
    logging.info("\n🔗 合并采样结果...")
    final_df = pd.concat([anomaly_sample, normal_sample])
    
    # 数据后处理
    logging.info("📝 生成最终数据格式...")
    final_df = final_df.sort_values(by=line_id_col)
    final_df = final_df.rename(columns={line_id_col: 'OriginalLineId'})
    final_df.insert(0, 'NewLineId', range(1, len(final_df)+1))
    
    # 保存结果
    logging.info("💾 保存采样结果...")
    final_df.to_csv(output_file, index=False)
    logging.info(f"✅ 采样完成！结果已保存至：{output_file}")
    
    # 最终统计
    logging.info("\n📊 最终样本统计：")
    logging.info(f"  - 异常样本：{final_anomaly_count} 条")
    logging.info(f"  - 正常样本：{len(normal_sample)} 条")
    logging.info(f"  - 总计：{len(final_df)} 条")

if __name__ == "__main__":
    # 示例调用
    optimized_sampling(
        log_type="HDFS",
        radio=10,
        scheme="anomaly_labels" #normal_based | anomaly_labels
    )
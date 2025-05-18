import numpy as np

# ==== 语义反馈策略调参 ====
GAMMA = 0.5  # 分歧阈值（低于此值表示语义差异大）
SIGMA = 0.85  # 一致阈值（高于此值表示高度一致）


def build_next_prompts(log_row, prev_results, sim_matrix, sim_avg):
    """
    根据语义相似度和前一轮推理结果，为下一轮构造新的 prompt。

    参数：
    - log_row: 当前日志（DataFrame 单行）
    - prev_results: 上一轮中三个模型的输出（包含 label, reason, score）
    - sim_matrix: 3x3 语义相似度矩阵
    - sim_avg: 当前轮次三者解释平均语义相似度

    返回：
    - next_prompts: [prompt_A, prompt_B, prompt_C]
    - strategy_flag: "hard", "soft", or "agree"
    """

    base_info = f"""模板：{log_row['EventTemplate']}
组件：{log_row['Component']}
等级：{log_row['Level']}
类型：{log_row['Type']}
节点：{log_row['Node']}
内容：{log_row['Content']}"""

    prompts = []

    if sim_avg >= SIGMA:
        # 共识程度很高，提示中可强调共识并加权投票
        strategy_flag = "agree"
        common_prefix = f"""你是一名日志分析专家。以下是系统记录的一条灰日志，请结合上下文判断是否异常并解释原因。
日志结构如下：
{base_info}

注意：多个模型判断结果较为一致，请你从中总结共识并给出你的最终判定。
"""
        prompts = [common_prefix] * 3

    elif sim_avg < GAMMA:
        # 分歧很大：强调你需要参考其它模型的全部 reasoning 内容
        strategy_flag = "hard"
        for i in range(3):
            others = [f"模型{chr(65 + j)}解释：{prev_results[j]['reason']}" for j in range(3) if j != i]
            full_prompt = f"""你是一名日志异常检测专家。请你重新判断以下日志是否异常，并优化自己的解释。
日志信息如下：
{base_info}

上轮你的判断为：{prev_results[i]['reason']}
以下是其它模型的解释：
{"；".join(others)}

请你结合他们的推理并修正自己的判断，输出JSON包含：label（0/1），reason（解释），score（0~1）。
"""
            prompts.append(full_prompt)

    else:
        # 中等相似度：鼓励自我优化+适度吸收他人信息
        strategy_flag = "soft"
        for i in range(3):
            others = [f"{chr(65 + j)}: {prev_results[j]['reason']}" for j in range(3) if j != i]
            full_prompt = f"""你是一名日志异常检测专家。请你再次判断以下日志是否异常，并优化你的解释。
日志结构如下：
{base_info}

上轮你的输出为：{prev_results[i]['reason']}
参考其它模型的观点：
{"；".join(others)}

请重新生成解释并输出 JSON，包含字段 label、reason、score。
"""
            prompts.append(full_prompt)

    return prompts, strategy_flag


# ✅ 测试入口
if __name__ == "__main__":
    import pandas as pd

    # 构造一个测试样本行
    test_log = {
        "EventTemplate": "ciod failed to read message prefix on control stream",
        "Component": "APP",
        "Level": "FATAL",
        "Type": "RAS",
        "Node": "R37-M0-N0-I:J18-U01",
        "Content": "ciod failed to read message prefix on control stream..."
    }

    # 模拟上一轮结果
    results = [
        {"label": 1, "reason": "组件 APP 通信失败，严重错误", "score": 0.92},
        {"label": 1, "reason": "FATAL 错误表明通信断裂", "score": 0.88},
        {"label": 1, "reason": "消息前缀读取失败，系统异常", "score": 0.91}
    ]

    # 构造相似度矩阵和平均相似度
    sim_matrix = np.array([
        [1.00, 0.87, 0.81],
        [0.87, 1.00, 0.85],
        [0.81, 0.85, 1.00]
    ])
    sim_avg = (0.87 + 0.81 + 0.85) / 3

    # 调用
    prompts, flag = build_next_prompts(test_log, results, sim_matrix, sim_avg)
    print(f"\n🧠 使用反馈策略：{flag}")
    for i, p in enumerate(prompts):
        print(f"\nAgent {chr(65 + i)} Prompt:\n{p[:300]}...")  # 截断显示

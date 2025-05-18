import numpy as np


def weighted_vote(results, sim_matrix, alpha=0.5):
    """
    对三个模型的输出进行置信+一致性加权投票。

    参数：
        results: List[dict]，每个元素是模型输出，包含 label（0/1），score（置信度），reason（解释）
        sim_matrix: np.ndarray，语义相似度矩阵（3x3）
        alpha: float，平衡因子（默认0.5）

    返回：
        final_label: 最终投票标签（0或1）
        label_score_map: 每个类别得分
    """
    labels = [r['label'] for r in results]
    scores = [r['score'] for r in results]

    # === 计算一致性分数（sim_avg） ===
    consistency_scores = []
    for i in range(3):
        others = [j for j in range(3) if j != i]
        sim_sum = sum(sim_matrix[i][j] for j in others)
        sim_avg = sim_sum / 2  # 另外两个的平均
        consistency_scores.append(sim_avg)

    # === 计算权重（wi） ===
    weights = []
    for i in range(3):
        wi = alpha * scores[i] + (1 - alpha) * consistency_scores[i]
        weights.append(wi)

    # === 加权计票 ===
    vote_score = {0: 0.0, 1: 0.0}
    for i in range(3):
        vote_score[labels[i]] += weights[i]

    # === 决策输出 ===
    final_label = max(vote_score, key=vote_score.get)
    return final_label, vote_score


# ✅ 单元测试
if __name__ == "__main__":
    mock_results = [
        {"label": 1, "score": 0.9, "reason": "异常等级高，内容严重"},
        {"label": 1, "score": 0.8, "reason": "系统FATAL，内容严重"},
        {"label": 0, "score": 0.7, "reason": "疑似正常"}
    ]

    mock_sim_matrix = np.array([
        [1.0, 0.85, 0.55],
        [0.85, 1.0, 0.60],
        [0.55, 0.60, 1.0]
    ])

    final, detail = weighted_vote(mock_results, mock_sim_matrix)
    print("\n🗳️ 最终标签：", final)
    print("📊 投票得分：", detail)

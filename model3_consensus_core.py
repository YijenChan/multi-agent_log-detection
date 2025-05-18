import time
from model3_agent1 import model3_agent_a_infer
from model3_agent2 import model3_agent_b_infer
from model3_agent3 import model3_agent_c_infer
from model3_similarity_utils import compute_similarity_matrix
from model3_feedback_utils import build_next_prompts
from model3_vote_utils import weighted_vote

MAX_ROUNDS = 3

def consensus_inference(row):
    """
    对单条日志执行多轮协同推理。
    返回: (final_label, consensus_flag, metadata)
    """
    history = []
    prompts = ["", "", ""]  # 初始提示为空

    for round_id in range(1, MAX_ROUNDS + 1):
        print(f"\n🌀 第 {round_id} 轮协同推理...")

        # === 推理调用 ===
        agents = [model3_agent_a_infer, model3_agent_b_infer, model3_agent_c_infer]
        results = []

        for i, agent in enumerate(agents):
            try:
                result = agent(row, prompt_override=prompts[i]) if round_id > 1 else agent(row)
                if result is None:
                    raise ValueError("空返回")
            except Exception as e:
                print(f"❌ 模型 {chr(65+i)} 推理失败: {e}")
                result = {"label": -1, "reason": "调用失败", "score": 0.0}
            results.append(result)

        labels = [r["label"] for r in results]
        reasons = {k: r["reason"] for k, r in zip(["A", "B", "C"], results)}
        scores = [r["score"] for r in results]

        print("🧾 当前推理标签：", labels)
        print("🗣️ 当前解释摘要：", list(reasons.values()))

        # === 共识判断 ===
        if all(l == labels[0] and l in [0, 1] for l in labels):
            print("✅ 标签完全一致，直接输出")
            return labels[0], "HARD", {
                "round": round_id,
                "reasons": reasons,
                "scores": scores,
                "method": "初轮一致"
            }

        # === 语义相似度分析 ===
        sim_avg, sim_matrix = compute_similarity_matrix(reasons)
        print(f"🔗 平均语义相似度 Sim_avg = {sim_avg:.3f}")

        valid_labels = all(l in [0, 1] for l in labels)
        if sim_avg >= 0.85 and valid_labels:
            label_final, vote_map = weighted_vote(results, sim_matrix)
            print("🤝 高度相似，共识投票输出")
            return label_final, "WEAK", {
                "round": round_id,
                "reasons": reasons,
                "scores": scores,
                "vote_map": vote_map,
                "method": "加权投票"
            }

        # === 准备下一轮 ===
        if round_id < MAX_ROUNDS:
            prompts, strategy_flag = build_next_prompts(row, results, sim_matrix, sim_avg)
            print(f"🛠️ 使用提示策略：{strategy_flag}，准备进入下一轮")
            time.sleep(1)

        history.append({"labels": labels, "reasons": reasons, "scores": scores})

    # === 达到最大轮次仍未共识 ===
    print("⚠️ 达到最大轮数仍未收敛")
    return None, "FAIL", {"round": MAX_ROUNDS, "history": history}


# ✅ 单元测试入口
if __name__ == "__main__":
    import pandas as pd

    test_path = "解析后的数据集.csv"
    df = pd.read_csv(test_path)
    row = df.iloc[10]

    label, status, detail = consensus_inference(row)
    print("\n🧾 最终共识标签:", label)
    print("📌 共识状态:", status)
    print("📊 明细信息:", detail)

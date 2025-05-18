from sentence_transformers import SentenceTransformer, util
import numpy as np

# === 加载嵌入模型（建议一次性加载） ===
model_name = 'all-MiniLM-L6-v2'  # SBERT 轻量版本
sbert = SentenceTransformer(model_name)


def compute_similarity_matrix(reasons: dict):
    """
    计算三条解释性文本之间的两两余弦相似度矩阵，以及平均语义相似度 Sim_avg。
    :param reasons: dict，如 {'A': '...', 'B': '...', 'C': '...'}
    :return: Sim_avg(float), 相似度矩阵(np.ndarray)
    """
    names = list(reasons.keys())
    texts = [reasons[k] for k in names]

    embeddings = sbert.encode(texts, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # 提取三对（AB, AC, BC）相似度并平均
    sim_ab = sim_matrix[0, 1]
    sim_ac = sim_matrix[0, 2]
    sim_bc = sim_matrix[1, 2]
    sim_avg = (sim_ab + sim_ac + sim_bc) / 3

    return float(sim_avg), sim_matrix


# ✅ 模块测试入口
if __name__ == "__main__":
    sample_reasons = {
        'A': "该日志为FATAL级别，内核组件出现严重错误。",
        'B': "日志等级为FATAL，说明系统存在致命故障。",
        'C': "此日志等级严重，可能引发系统崩溃，属于异常日志。"
    }
    sim_avg, matrix = compute_similarity_matrix(sample_reasons)

    print("\n📐 相似度矩阵：")
    print(np.round(matrix, 3))
    print(f"\n🔗 平均语义相似度 Sim_avg: {sim_avg:.4f}")

import pandas as pd
import json
from model2_1_CS_A import get_model_A_result
from model2_2_CT_B import get_model_B_score
import time

# ==== 参数配置 ====
CSV_PATH = "解析后的数据集.csv"
OUTPUT_PATH = "融合器评估结果.json"
ALPHA = 0.3  # 一致性阈值 Δ
BETA = 0.7  # 接受可信度阈值 G
MAX_RETRY = 3  # 最大重试次数
SLEEP_SECONDS = 0.1  # 调用间隔

# ==== 加载数据 ====
df = pd.read_csv(CSV_PATH)
results = []

# ==== 主处理流程 ====
for idx, row in df.iterrows():
    print(f"\n🟦 [第 {idx + 1}/{len(df)} 条日志]")

    # === 模型 A 重试 ===
    result_a = None
    for attempt in range(1, MAX_RETRY + 1):
        result_a = get_model_A_result(row)
        if result_a:
            break
        print(f"⚠️ 模型A 第 {attempt} 次尝试失败，重试中...")
        time.sleep(1)

    if not result_a:
        print("❌ 模型A连续失败，标记当前日志为处理失败")
        results.append({
            "index": idx,
            "status": "模型A失败",
            "log_row": row.to_dict()
        })
        continue

    # === 模型 B 重试 ===
    result_b = None
    for attempt in range(1, MAX_RETRY + 1):
        result_b = get_model_B_score(row, result_a)
        if result_b is not None:
            break
        print(f"⚠️ 模型B 第 {attempt} 次尝试失败，重试中...")
        time.sleep(1)

    if result_b is None:
        print("❌ 模型B连续失败，标记当前日志为处理失败")
        results.append({
            "index": idx,
            "status": "模型B失败",
            "model_A": result_a,
            "log_row": row.to_dict()
        })
        continue

    # === 处理结果 ===
    # 模型B可能返回float或dict
    if isinstance(result_b, dict):
        score_b = float(result_b.get("score", 0))
        reason_b = result_b.get("reason", "")
    else:
        score_b = float(result_b)
        reason_b = ""

    score_a = float(result_a["score"])
    delta = abs(score_a - score_b)
    fusion_score = (score_a + score_b) / 2

    # 判断是否采纳标签
    accept_flag = (delta < ALPHA and fusion_score > BETA)

    # 分类判断
    if delta >= ALPHA:
        final_label = "灰日志"
    elif fusion_score > BETA and result_a["label"] == 1:
        final_label = "黑日志"
    elif fusion_score > BETA and result_a["label"] == 0:
        final_label = "白日志"
    else:
        final_label = "灰日志（需重判）"

    # === 控制台输出状态 ===
    print(f"✅ CS: {score_a:.2f} | CT: {score_b:.2f} | Δ: {delta:.2f} | G: {fusion_score:.2f}")
    print(f"📌 最终分类：{final_label} | 标签采纳：{accept_flag}")

    # === 写入结果 ===
    results.append({
        "index": idx,
        "fusion_score": round(fusion_score, 3),
        "delta": round(delta, 3),
        "accept_flag": accept_flag,
        "classification": final_label,
        "model_A": result_a,
        "model_B_score": score_b,
        "model_B_reason": reason_b,
        "log_row": row.to_dict()
    })

    time.sleep(SLEEP_SECONDS)

# ==== 输出保存 ====
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n🎉 所有处理完成，共计 {len(results)} 条，结果已保存至：{OUTPUT_PATH}")


# ==== 统计采纳结果 ====
tp = fp = fn = tn = 0
accepted_count = 0
black_accepted = 0
white_accepted = 0

for r in results:
    if r.get("accept_flag"):
        accepted_count += 1
        pred = int(r["model_A"]["label"])              # 模型预测标签
        true = int(r["log_row"]["BinaryLabel"])        # 真实标签

        if pred == 1 and true == 1:
            tp += 1
            black_accepted += 1
        elif pred == 1 and true == 0:
            fp += 1
            black_accepted += 1
        elif pred == 0 and true == 1:
            fn += 1
            white_accepted += 1
        elif pred == 0 and true == 0:
            tn += 1
            white_accepted += 1

# ==== 计算指标 ====
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# ==== 控制台打印 ====
print("\n📊 【已采纳样本统计】")
print(f"✅ 黑日志数量（预测为1）：{black_accepted}")
print(f"✅ 白日志数量（预测为0）：{white_accepted}")
print(f"🎯 精确度 Precision：{precision:.3f}")
print(f"🎯 召回率 Recall：{recall:.3f}")
print(f"🎯 F1 分数 F1-score：{f1_score:.3f}")

# ==== 灰日志池导出（实验模式）====
# 说明：将以下两类日志送入灰日志池：
# 1）融合器标记为灰日志；
# 2）融合器采纳了错误标签（预测 ≠ 真实标签）

gray_pool = []

for r in results:
    cls = r.get("classification", "")
    true_label = int(r["log_row"].get("BinaryLabel", -1))
    pred_label = int(r["model_A"].get("label", -1))

    # 条件1：被判为灰日志
    if "灰日志" in cls:
        gray_pool.append(r["log_row"])

    # 条件2：虽然采纳了标签但结果错误（仅在已知标签时成立）
    elif r.get("accept_flag") and pred_label != true_label:
        gray_pool.append(r["log_row"])

# 保存灰日志池
if gray_pool:
    gray_df = pd.DataFrame(gray_pool)
    gray_path = "融合器筛选出的灰日志.csv"
    gray_df.to_csv(gray_path, index=False)
    print(f"\n🟨 灰日志池导出完成，共 {len(gray_pool)} 条，路径：{gray_path}")
else:
    print("\n✅ 未发现灰日志或误判样本，无需导出")

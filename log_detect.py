import pandas as pd
import numpy as np
import json
import time

from model2_1_CS_A import get_model_A_result
from model2_2_CT_B import get_model_B_score
from model3_consensus_core import consensus_inference

# ==== 配置参数 ====
INPUT_PATH = "解析后的数据集.csv"
OUTPUT_PATH = "检测结果.json"
GRAY_POOL_PATH = "灰日志池数据.csv"

ALPHA = 0.3  # 一致性阈值 Δ
BETA = 0.7   # 接受可信度阈值 G
MAX_RETRY = 3

# ==== 类型转换器 ====
def convert_to_builtin_type(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# ==== 加载数据 ====
df = pd.read_csv(INPUT_PATH)
results = []
gray_logs = []

# ==== 主循环处理 ====
for idx, row in df.iterrows():
    print(f"\n🔍 正在处理第 {idx + 1}/{len(df)} 条日志...")

    row_dict = {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in row.to_dict().items()}

    # === 模型 A 推理 ===
    result_a = None
    for _ in range(MAX_RETRY):
        result_a = get_model_A_result(row)
        if result_a:
            break
        time.sleep(0.5)

    if not result_a:
        print("❌ 模型A连续失败，跳过")
        results.append({"index": idx, "status": "模型A失败", "log": row_dict})
        continue

    # === 模型 B 推理 ===
    result_b = None
    for _ in range(MAX_RETRY):
        result_b = get_model_B_score(row, result_a)
        if result_b is not None:
            break
        time.sleep(0.5)

    if result_b is None:
        print("❌ 模型B连续失败，跳过")
        results.append({
            "index": idx,
            "status": "模型B失败",
            "model_A": result_a,
            "log": row_dict
        })
        continue

    # === 分数与融合 ===
    score_a = float(result_a["score"])
    score_b = float(result_b["score"]) if isinstance(result_b, dict) else float(result_b)
    delta = abs(score_a - score_b)
    fusion_score = (score_a + score_b) / 2
    label_a = result_a["label"]
    is_gray = (delta >= ALPHA) or (fusion_score <= BETA)

    fusion_label = "灰日志"
    if not is_gray:
        fusion_label = "黑日志" if label_a == 1 else "白日志"

    print(f"✅ 融合器判定：{fusion_label}（Δ={delta:.2f}, G={fusion_score:.2f}）")

    # === 共识处理 ===
    consensus_info = None
    if "灰" in fusion_label:
        label_final, flag, detail = consensus_inference(row)
        consensus_info = {
            "final_label": label_final,
            "status": flag,
            "detail": detail
        }

        if flag == "FAIL":
            gray_logs.append(row_dict)
        else:
            fusion_label += f"（共识修正为 {label_final}）"

    # === 汇总结果 ===
    results.append({
        "index": idx,
        "log": row_dict,
        "model_A": result_a,
        "model_B_score": round(score_b, 3),
        "delta": round(delta, 3),
        "fusion_score": round(fusion_score, 3),
        "fusion_label": fusion_label,
        "consensus": consensus_info
    })

# ==== 保存结果 JSON ====
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=convert_to_builtin_type)
print(f"\n✅ 检测完成，结果保存至：{OUTPUT_PATH}")

# ==== 灰日志导出 ====
if gray_logs:
    pd.DataFrame(gray_logs).to_csv(GRAY_POOL_PATH, index=False)
    print(f"🟨 共导出灰日志 {len(gray_logs)} 条 → {GRAY_POOL_PATH}")
else:
    print("🎉 所有灰日志已成功共识，无需导出灰日志池")

# ==== 精度评估 ====
tp = fp = fn = tn = 0
for r in results:
    log = r.get("log", {})
    true = int(log.get("BinaryLabel", -1))
    pred = r["model_A"]["label"]

    if r.get("fusion_label", "").startswith("黑"):
        if true == 1:
            tp += 1
        else:
            fp += 1
    elif r.get("fusion_label", "").startswith("白"):
        if true == 0:
            tn += 1
        else:
            fn += 1

precision = tp / (tp + fp) if (tp + fp) else 0
recall    = tp / (tp + fn) if (tp + fn) else 0
f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("\n📊 【检测结果评估】")
print(f"✔️ TP（真阳性）: {tp}")
print(f"✔️ FP（假阳性）: {fp}")
print(f"✔️ FN（漏报）  : {fn}")
print(f"✔️ TN（真阴性）: {tn}")
print(f"\n🎯 精确度 Precision: {precision:.3f}")
print(f"🎯 召回率 Recall   : {recall:.3f}")
print(f"🎯 F1 分数 F1-score: {f1_score:.3f}")

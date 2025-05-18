import openai
import json
import time

def model3_agent_a_infer(row, prompt_override=None, max_retry=3):
    """
    使用 GPT-3.5 对日志记录进行分类 + 解释推理。
    自动校验格式，必要时多轮重试。
    """
    openai.api_key = '模型A API key'
    openai.api_base = "模型A代理"

    # === 构造默认提示词（支持外部覆盖）===
    if prompt_override:
        prompt = prompt_override
    else:
        prompt = f"""
你是一名日志异常检测专家，请判断以下日志是否异常，并简要解释原因：
模板：{row['EventTemplate']}
组件：{row['Component']}
等级：{row['Level']}
类型：{row['Type']}
节点：{row['Node']}
内容：{row['Content']}

⚠️ 输出格式必须严格为 JSON，包含以下字段：
- "label": 只能是 0 或 1（只能返回0或1，不能返回其它类型描述。0表示“正常”，1表示“异常”）
- "reason": 不超过200字的中文解释原因
- "score": 置信度，0~1之间的小数，表示对结果的自信程度

示例输出（仅此格式）：
{{"label": 1, "reason": "日志等级为FATAL，存在严重错误", "score": 0.92}}
        """.strip()

    # === 多轮重试调用 ===
    for attempt in range(1, max_retry + 1):
        try:
            response = openai.ChatCompletion.create(
                model="模型Aname",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=20
            )

            content = response.choices[0].message["content"].strip()
            parsed = json.loads(content)

            # === 强制格式校验 ===
            label_raw = parsed.get("label")
            reason = parsed.get("reason", "").strip()
            score = float(parsed.get("score", 0.0))

            # === label 容错处理 ===
            if isinstance(label_raw, str):
                norm = label_raw.lower()
                if "异常" in norm or "abnormal" in norm:
                    label = 1
                elif "正常" in norm or "normal" in norm:
                    label = 0
                else:
                    label = int(label_raw)
            else:
                label = int(label_raw)

            # 校验通过
            if label in [0, 1] and 0.0 <= score <= 1.0:
                return {"label": label, "reason": reason, "score": score}
            else:
                raise ValueError("解析结果不合法")

        except Exception as e:
            print(f"⚠️ 模型A 第 {attempt} 次尝试失败: {e}")
            time.sleep(1)

    # 所有尝试失败
    print("❌ [模型A调用失败] 多轮尝试未成功")
    return None

# ✅ 单元测试入口
if __name__ == "__main__":
    import pandas as pd
    test_path = "解析后的数据集.csv"
    df = pd.read_csv(test_path)
    row = df.iloc[0]
    result = model3_agent_a_infer(row)
    print("\n🎯 模型A 推理结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

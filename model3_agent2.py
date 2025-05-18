import openai
import json
import re
import time

def extract_json(text):
    """
    从模型输出中提取 JSON 字符串
    """
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    return match.group(0) if match else None

def model3_agent_b_infer(row, prompt_override=None, max_retry=3):
    """
    使用 GPT-4o 推理日志异常。返回 dict 包含 label, reason, score
    """
    openai.api_key = '模型B API key'
    openai.api_base = "模型B代理"

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

例如：
{{"label": 1, "reason": "日志等级为FATAL，存在严重错误", "score": 0.92}}
        """.strip()

    for attempt in range(1, max_retry + 1):
        try:
            response = openai.ChatCompletion.create(
                model="模型Bname",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=20
            )
            raw_content = response.choices[0].message["content"].strip()
            json_str = extract_json(raw_content)
            if not json_str:
                raise ValueError("未能提取出合法 JSON 格式")

            parsed = json.loads(json_str)

            # === label 容错解析 ===
            label_raw = parsed.get("label")
            reason = parsed.get("reason", "").strip()
            score = float(parsed.get("score", 0.0))

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

            if label in [0, 1] and 0.0 <= score <= 1.0:
                return {"label": label, "reason": reason, "score": score}
            else:
                raise ValueError("字段值超出预期范围")

        except Exception as e:
            print(f"⚠️ 模型B 第 {attempt} 次尝试失败: {e}")
            time.sleep(1)

    print("❌ [模型B调用失败] 多轮尝试未成功")
    return None

# ✅ 单元测试
if __name__ == "__main__":
    import pandas as pd
    test_path = "解析后的数据集.csv"
    df = pd.read_csv(test_path)
    row = df.iloc[0]
    result = model3_agent_b_infer(row)
    print("\n🎯 模型B 推理结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

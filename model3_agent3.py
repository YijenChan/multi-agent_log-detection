import openai
import json
import re
import time

def extract_json(text):
    """
    从模型返回中提取 JSON（处理 markdown 或额外说明文本）
    """
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    return match.group(0) if match else None

def model3_agent_c_infer(row, prompt_override=None, max_retry=3):
    """
    使用 DeepSeek API 模拟 GPT-4 级别模型，返回异常检测结果。
    支持 prompt_override，用于多轮协同推理。
    返回：dict，包括 label, reason, score
    """
    openai.api_key = '模型C API key'
    openai.api_base = "模型C代理"

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

示例：
{{"label": 1, "reason": "日志等级为FATAL，表示系统出现严重错误", "score": 0.92}}
        """.strip()

    for attempt in range(1, max_retry + 1):
        try:
            response = openai.ChatCompletion.create(
                model="模型Cname",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=20
            )

            raw_content = response.choices[0].message["content"].strip()
            json_str = extract_json(raw_content)
            if not json_str:
                raise ValueError("未能提取合法 JSON 格式")

            parsed = json.loads(json_str)

            # === 容错解析 label ===
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

            if label not in [0, 1] or not (0.0 <= score <= 1.0):
                raise ValueError("label 或 score 不在合法范围内")

            return {"label": label, "reason": reason, "score": score}

        except Exception as e:
            print(f"⚠️ 模型C 第 {attempt} 次尝试失败: {e}")
            time.sleep(1)

    print("❌ [模型C调用失败] 多轮尝试均失败")
    return None

# ✅ 单元测试
if __name__ == "__main__":
    import pandas as pd
    test_path = "解析后的数据集.csv"
    df = pd.read_csv(test_path)
    row = df.iloc[0]

    result = model3_agent_c_infer(row)
    print("\n🎯 模型C 推理结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

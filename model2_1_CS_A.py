import openai
import json

import openai
import json
import time

def get_model_A_result(row, max_retry=3):
    openai.api_key = '学生模型API key'
    openai.api_base = "学生模型代理"

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

示例输出：
{{"label": 1, "reason": "日志等级为FATAL，表示系统出现严重错误", "score": 0.92}}

⚠️ 请不要输出 markdown 包裹，不要添加解释说明，仅输出 JSON。
    """.strip()

    for attempt in range(1, max_retry + 1):
        try:
            response = openai.ChatCompletion.create(
                model="学生模型name",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=20
            )
            content = response.choices[0].message["content"].strip()

            # 提取 JSON 内容
            json_str = content
            if json_str.startswith("```"):
                json_str = json_str.strip("`").strip()
                if json_str.lower().startswith("json"):
                    json_str = json_str[4:].strip()
            parsed = json.loads(json_str)

            # 处理 label 容错
            label_raw = parsed["label"]
            if isinstance(label_raw, str):
                if "异常" in label_raw.lower() or "abnormal" in label_raw.lower():
                    label = 1
                elif "正常" in label_raw.lower() or "normal" in label_raw.lower():
                    label = 0
                else:
                    label = int(label_raw)
            else:
                label = int(label_raw)

            score = float(parsed["score"])
            if label not in [0, 1] or not (0 <= score <= 1):
                raise ValueError("非法label或score")

            return {
                "label": label,
                "reason": parsed["reason"].strip(),
                "score": score
            }

        except Exception as e:
            print(f"⚠️ 模型A 第 {attempt} 次尝试失败: {e}")
            time.sleep(1)

    print("❌ 调用模型 A 失败：已重试多次")
    return None

# ✅ 测试入口（不会影响主流程调用）
if __name__ == "__main__":
    import pandas as pd

    # 读取一条样本数据进行测试
    test_path = "解析后的数据集.csv"
    df = pd.read_csv(test_path)
    sample_row = df.iloc[0]

    result = get_model_A_result(sample_row)
    print("✅ 测试返回：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

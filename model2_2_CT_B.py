import openai
import json
import openai
import json
import time
import re

def extract_json(text):
    """从模型返回中提取 JSON 内容（去除 markdown）"""
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    return match.group(0) if match else None

def get_model_B_score(row, model_a_result, max_retry=3):
    openai.api_key = '教师模型API key'
    openai.api_base = "教师模型代理"

    prompt = f"""
你是一名高级日志分析专家，请你评估另一位分析师（模型A）的判断是否可信。
以下是原始日志信息（供你参考）：
模板：{row['EventTemplate']}
组件：{row['Component']}
等级：{row['Level']}
类型：{row['Type']}
节点：{row['Node']}
内容：{row['Content']}

模型A的判断如下（JSON格式）：
{json.dumps(model_a_result, ensure_ascii=False)}

请你仅输出以下格式内容：
{{"score": 0.85}}

⚠️ 注意：
- 请不要添加任何解释说明
- "score": 置信度，0~1之间的小数，表示对模型A检测结果的信任程度
- 只输出 JSON 格式，禁止 markdown 包裹
""".strip()

    for attempt in range(1, max_retry + 1):
        try:
            response = openai.ChatCompletion.create(
                model="教师模型name",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=20
            )
            raw_content = response.choices[0].message["content"].strip()
            json_str = extract_json(raw_content)
            if not json_str:
                raise ValueError("返回格式非 JSON")

            parsed = json.loads(json_str)
            score = float(parsed["score"])

            # 分数合法性判断
            if not (0 <= score <= 1):
                raise ValueError("score 超出合法范围")

            return {"score": score}

        except Exception as e:
            print(f"⚠️ 模型B 第 {attempt} 次失败：{e}")
            time.sleep(1)

    print("❌ 调用模型 B 失败：已重试多次")
    return None

# ✅ 测试入口（独立测试用，不影响主流程）
if __name__ == "__main__":
    import pandas as pd

    # 读取测试数据
    test_path = "解析后的数据集.csv"
    df = pd.read_csv(test_path)
    sample_row = df.iloc[0]

    # 模拟模型A的返回结果（测试用）
    mock_a_result = {
        "reason": "ciod在通信时无法读取消息前缀，可能导致应用错误",
        "label": 1,
        "score": 0.9
    }

    result = get_model_B_score(sample_row, mock_a_result)
    print("✅ 模型B返回：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

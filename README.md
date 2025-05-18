# Multi-Agent Log Detection

A multi-agent intelligent system for abnormal log detection and reasoning, combining ensemble judgment, collaborative feedback, and semantic consensus.

## 📌 Project Overview

This repository implements a multi-agent log anomaly detection framework based on LLMs. It integrates:
- Multiple expert agents (Agent1/2/3) for primary anomaly classification.
- Confidence fusion to determine log label (black/white/grey).
- Consensus mechanism for collaborative interpretation of grey logs.

It supports datasets like BGL and HDFS, and generates both labels and natural language explanations.

---

## 📁 Folder Structure

```

multi-agent\_log-detection/
├── log\_detect.py                 # Main pipeline entry
├── Confidence Fusion.py         # Confidence-based label integration
├── model2\_1\_CS\_A.py             # Model A: initial classification
├── model2\_2\_CT\_B.py             # Model B: confidence evaluator
├── model3\_agent\[1|2|3].py       # Three expert agents for multi-perspective reasoning
├── model3\_feedback\_utils.py     # Feedback adjustment for gray logs
├── model3\_similarity\_utils.py   # Semantic similarity functions
├── model3\_vote\_utils.py         # Voting logic for consensus
├── model3\_consensus\_core.py     # Multi-round consensus orchestration
├── DATASET\_TEST.csv             # Sample dataset for testing
├── BGL\_process.csv              # Preprocessed BGL dataset (partial)
├── HDFS\_process.csv             # Preprocessed HDFS dataset (partial)
└── 解析脚本/                     # Dataset processing scripts

````

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install openai pandas
````

2. **Edit API Keys**:
   Make sure your agent files (`model3_agent*.py`) include valid `openai.api_key` and `api_base`.

3. **Execute main pipeline**:

   ```bash
   python log_detect.py
   ```

   Logs will be processed, labeled, and exported to `test_log_detect_results.json`.

---

## 🧠 Features

* 🧩 Modular agent design: easy to extend/replace models.
* ⚖️ Fusion-based confidence scoring.
* 🌀 Dynamic gray log consensus with GPT-based explanations.
* 📈 Evaluation metrics: precision, recall, F1.
* ✅ Support for multi-round decision making.

---

## 📊 Sample Output

* Log type prediction: ✅ 黑日志 / 白日志 / 灰日志
* Multi-agent explanations:

  ```
  🗣️ ["日志等级为FATAL，可能表示系统严重故障", 
      "断言失败指示关键组件异常", 
      "内容涉及Microloader，属于内核级错误"]
  ```
* Export: `test_log_detect_results.json`, `test_log_grey_pool.csv`

---

## 📁 Datasets

* `BGL_process.csv` and `HDFS_process.csv`: processed samples for demo purposes.
* Full datasets too large for GitHub are available upon request or via your local `F:/muti_agents_log/`.

---

## 🛠 Future Work

* Add GUI for interactive anomaly analysis.
* Support fine-tuning custom LLMs.
* Integrate structured log parsing pipeline.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgments

* Inspired by multi-agent decision theory and LLM-based reasoning techniques.
* Powered by OpenAI and DeepSeek APIs.

```

---

✅ 你可以现在点击右上角的 **"Commit changes"** 完成 README 的添加。

如果你有特定的展示截图、论文链接或数据源网址，也可以附加在最后的章节中「📄 Acknowledgments」或新建「📷 Examples」部分。需要我继续扩展也随时告诉我！
```

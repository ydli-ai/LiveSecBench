# LiveSecBench

<div align="center">

**A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.02366-b31b1b.svg)](https://arxiv.org/abs/2511.02366)

 [English](./README_EN.md) | [简体中文](./README.md)

</div>

LiveSecBench 是一个面向中文场景的大模型安全评测基准。框架结合动态题库、模型对战与客观评分流程，可在伦理、合法性、事实性、隐私、对抗鲁棒与推理安全等核心维度持续追踪模型表现。完整方法论请参见论文 [LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context](https://arxiv.org/abs/2511.02366)。

## 微信交流群

欢迎扫码加入 LiveSecBench 微信交流群，获取最新评测动态与技术讨论：

![LiveSecBench 微信交流群](https://filecdn.intokentech.cn/public/weixin_group.png)

## 项目亮点

- 真实语境：题库覆盖多种中文安全场景，区分显性与隐性风险表达。
- 对战式评分：ELO 流程支持瑞士制 / 循环 / 随机配对，可选收敛检测提前停止。
- 任务溯源：评测任务、模型输出与 PK 结果统一落盘到 SQLite，便于复现与审计。
- 自动报告：生成排行榜、统计摘要与 Markdown 报告。
- 离线演示：Mock 脚本可在无 API Key 环境模拟完整流程，快速验证配置。

## 快速开始

### 环境要求
- Python 3.10+
- 8GB RAM / 10GB 磁盘空间
- 建议使用 `conda create -n livesecbench python=3.10 && conda activate livesecbench`

### 安装
```bash
git clone https://github.com/ydli-ai/LiveSecBench.git
cd LiveSecBench
python -m pip install -e .
```

### 配置环境变量
```bash
export OPENAI_API_KEY="your_openai_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

### 运行评测
```bash
python livesecbench/run_livesecbench.py --config livesecbench/configs/run_custom_safety_benchmark.yaml
```

### 运行测试
```bash
python -m pip install -e .[test]
pytest -v
pytest -k config_manager -v  # 仅验证配置解析等模块
```

### Mock 端到端演示
```bash
python scripts/run_mock_e2e.py
```
脚本会加载 `livesecbench/configs/mock_e2e.yaml`，Mock 所有 HTTP 请求，并把演示数据写入 `mock_results/`、`mock_history/`、`mock_records/` 以及 `data/mock_e2e.db`。

## 配置概览

- 主配置位于 `livesecbench/configs/run_custom_safety_benchmark.yaml`。
- 在 `models_to_test` 中为每个模型配置 API 信息（`base_url`、`api_key`、`model_id`、可选 `provider_ignore`），均支持 `env_var:` 引用环境变量。
- `question_selection` 可混合多个维度、版本或样本数量限制。
- `scoring_settings.model_based.elo` 控制配对策略、收敛检测、输出目录等参数。
- `judge_model_api` 指定裁判模型，默认 `deepseek-chat`。
- 详尽说明与最佳实践请参见 `docs/USER_GUIDE.md`。

## 结果产出

- `results/{日期}/`
  - `{月份}-models*.csv`：各维度综合得分与排名
  - `{月份}-stats*.csv`：评测统计摘要
  - `elo_results/{维度}/`：维度级 ELO 历史与 PK 详情（含 Excel）
  - `summary_report*.md`：自动生成的评测报告
- `data/livesecbench.db`
  - `model_outputs`：模型回答记录
  - `pk_results`：模型对战结果
  - `evaluation_tasks`：任务元信息

## 排行榜与文档

- 在线榜单：<https://livesecbench.intokentech.cn/>
- **[用户指南](./docs/USER_GUIDE.md)** - 详细的使用说明和配置指南
- **[架构文档](./docs/ARCHITECTURE.md)** - 系统架构和设计理念
- **[API 文档](./docs/API_DOCUMENTATION.md)** - 完整的 API 参考
- **[使用示例](./docs/EXAMPLES.md)** - 代码示例与最佳实践
- 论文与技术报告：<https://arxiv.org/abs/2511.02366>

## 项目结构

```
LiveSecBench/
├── livesecbench/
│   ├── core/
│   │   ├── run_model_answer.py   # 拉取模型回复
│   │   ├── run_scoring.py        # ELO 评分编排
│   │   ├── rank.py               # 排名聚合
│   │   ├── report.py             # Markdown 报告生成
│   │   └── task_manager.py       # 任务生命周期
│   ├── infra/
│   │   ├── config/               # ConfigManager、schema 校验
│   │   ├── storage/              # SQLiteStorage、缓存接口
│   │   ├── scoring/              # 配对策略、收敛检测
│   │   ├── batch_processor.py    # 批量调度
│   │   ├── cache_manager.py      # 输出缓存
│   │   └── http_client.py        # 带重试/限流的 HTTP 客户端
│   ├── scorers/
│   │   └── model_based_scorer.py # 基于裁判模型的评分逻辑
│   ├── question_set/             # 题库 JSON/CSV
│   ├── configs/                  # 运行配置样例（含自定义基准）
│   ├── utils/                    # 日志、环境变量加载等
│   └── run_livesecbench.py       # CLI 入口
├── docs/                         # 用户指南、架构、API 文档
├── scripts/
│   └── run_mock_e2e.py           # Mock 端到端脚本
├── tests/                        # pytest 用例
├── data/                         # SQLite 数据库
└── results/                      # 评测输出（按日期归档）
```

## 引用

```bibtex
@article{livesecbench,
  title={LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context},
  author={Yudong Li, Zhongliang Yang, Kejiang Chen, Wenxuan Wang, Tianxin Zhang, Sifang Wan, Kecheng Wang, Haitian Li, Xu Wang, Lefan Cheng, Youdan Yang, Baocheng Chen, Ziyu Liu, Yufei Sun, Liyan Wu, Wenya Wen, Xingchi Gu, Peiru Yang},
  year={2025},
}
```
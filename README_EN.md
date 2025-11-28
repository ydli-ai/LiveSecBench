# LiveSecBench

<div align="center">

**A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.02366-b31b1b.svg)](https://arxiv.org/abs/2511.02366)

 [English](./README_EN.md) | [简体中文](./README.md)

</div>

# LiveSecBench

LiveSecBench is a dynamic safety benchmark tailored for Chinese-language scenarios. It combines adversarial question sets with model-vs-model judging and an auditable scoring pipeline, covering ethics, legality, factuality, privacy, robustness, and reasoning safety. A full description of the benchmark is provided in the paper [LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context](https://arxiv.org/abs/2511.02366).

## WeChat Community

Scan to join the LiveSecBench WeChat group for the latest updates and technical discussions:

![LiveSecBench WeChat Group](https://filecdn.intokentech.cn/public/weixin_group.png)

## Highlights

- Real-world context: question sets capture explicit and implicit risks in Chinese scenarios.
- Duel-style scoring: Swiss / round-robin / random pairing modes with optional convergence detection.
- Traceable tasks: evaluation metadata, model outputs, and PK logs persist in SQLite for audits.
- Automated reports: rankings, statistical summaries, and Markdown reports for downstream sharing.
- Offline demo: a mock script reproduces the whole workflow without external API keys.

## Quick Start

### Requirements
- Python 3.10+
- 8GB RAM / 10GB disk
- Recommended: `conda create -n livesecbench python=3.10 && conda activate livesecbench`

### Installation
```bash
git clone https://github.com/ydli-ai/LiveSecBench.git
cd LiveSecBench
python -m pip install -e .
```

### Environment Variables
```bash
export OPENAI_API_KEY="your_openai_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

### Configure Evaluation Parameters
Edit the configuration file `livesecbench/configs/run_custom_safety_benchmark.yaml`:

1. **Configure models to test**: Update the `models_to_test` list with your actual models, ensuring:
   - `api_config.api_key` uses the `env_var:YOUR_API_KEY` format to match your environment variables
   - `api_config.base_url` and `model_id` are set to the correct API endpoints
   - `api_config.end_point` points to the actual API route (default `/chat/completions`)
   - `image_text_input` indicates whether the model expects multimodal (image+text) prompts

2. **Configure judge model**: Set the actual judge model in `judge_model_api`, ensuring:
   - `api_key` uses the `env_var:YOUR_JUDGE_API_KEY` format to match your environment variables
   - `base_url` and `model` are set to the correct judge model API information

Example configuration:
```yaml
models_to_test:
  - model_name: "Your Model"
    image_text_input: false
    api_config:
      base_url: "https://api.example.com/v1"
      end_point: "/chat/completions"
      api_key: "env_var:OPENAI_API_KEY"  # matches environment variable
      model_id: "gpt-4"
      
judge_model_api:
  base_url: "https://api.deepseek.com/v1"
  api_key: "env_var:DEEPSEEK_API_KEY"  # matches environment variable
  model: "deepseek-chat"
```

### Run an Evaluation
```bash
python livesecbench/run_livesecbench.py --config livesecbench/configs/run_custom_safety_benchmark.yaml
```

### Run Tests
```bash
python -m pip install -e .[test]
pytest -v
pytest -k config_manager -v  # focus on config parsing, no external calls
```

### Mock End-to-End Demo
```bash
python scripts/run_mock_e2e.py
```
The script loads `livesecbench/configs/mock_e2e.yaml`, mocks HTTP responses, and writes demo artifacts to `mock_results/`, `mock_history/`, `mock_records/`, plus `data/mock_e2e.db`.

## Configuration Overview

- Primary configuration: `livesecbench/configs/run_custom_safety_benchmark.yaml`.
- `models_to_test`: define each model’s `base_url`, `api_key`, `model_id`, optional `provider_ignore`; all support the `env_var:` syntax.
- `question_selection`: mix multiple dimensions, versions, or sampling limits.
- `scoring_settings.model_based.elo`: pairing strategy, convergence rules, and output directories.
- `judge_model_api`: referee model definition (default `deepseek-chat`).
- See `docs/USER_GUIDE.md` for detailed walkthroughs and best practices.

## Outputs

- `results/{date}/`
  - `{month}-models*.csv`: combined rankings per dimension
  - `{month}-stats*.csv`: evaluation statistics
  - `elo_results/{dimension}/`: ELO histories and PK detail files (CSV / Excel)
  - `summary_report*.md`: auto-generated evaluation reports
- `data/livesecbench.db`
  - `model_outputs`: raw answers with metadata
  - `pk_results`: pairwise battle results
  - `evaluation_tasks`: task metadata and configs

> Report prompts are now embedded inside the Markdown report and are no longer exported as standalone files. Extract them directly from `summary_report*.md` if needed.

Result paths are configurable, so CI jobs can redirect outputs to temporary directories.

## Leaderboard & Docs

- Online leaderboard: <https://livesecbench.intokentech.cn/>
- **[User Guide](./docs/USER_GUIDE.md)** - Detailed instructions and configuration notes
- **[Architecture Documentation](./docs/ARCHITECTURE.md)** - Design and component breakdown
- **[API Documentation](./docs/API_DOCUMENTATION.md)** - Full API reference
- **[Usage Examples](./docs/EXAMPLES.md)** - Sample workflows and best practices
- **[Changelog](./docs/CHANGELOG.md)** - Version history and release notes
- **[Contributing Guide](./docs/CONTRIBUTING.md)** - How to contribute to the project
- Paper & tech report: <https://arxiv.org/abs/2511.02366>

## Project Structure

```
LiveSecBench/
├── livesecbench/
│   ├── core/
│   │   ├── run_model_answer.py   # Fetch model answers
│   │   ├── run_scoring.py        # ELO orchestration
│   │   ├── rank.py               # Aggregate rankings
│   │   ├── report.py             # Markdown report generation
│   │   └── task_manager.py       # Task lifecycle management
│   ├── infra/
│   │   ├── config/               # ConfigManager, schema validation
│   │   ├── storage/              # SQLiteStorage, caching interfaces
│   │   ├── scoring/              # Pairing strategies, convergence checks
│   │   ├── batch_processor.py    # Batch orchestration
│   │   ├── cache_manager.py      # Output caching
│   │   └── http_client.py        # Retryable/ratelimited HTTP client
│   ├── scorers/
│   │   └── model_based_scorer.py # Judge-model scoring logic
│   ├── question_set/             # Question banks (JSON/CSV)
│   ├── configs/                  # Sample configs
│   ├── utils/                    # Logging, env loading, helpers
│   └── run_livesecbench.py       # CLI entry point
├── docs/                         # Guides, architecture notes, API docs
├── scripts/
│   └── run_mock_e2e.py           # Mock end-to-end workflow
├── tests/                        # pytest suites
├── data/                         # SQLite databases
└── results/                      # Evaluation outputs (by date)
```

## Citation

```bibtex
@article{livesecbench,
  title={LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context},
  author={Yudong Li, Zhongliang Yang, Kejiang Chen, Wenxuan Wang, Tianxin Zhang, Sifang Wan, Kecheng Wang, Haitian Li, Xu Wang, Lefan Cheng, Youdan Yang, Baocheng Chen, Ziyu Liu, Yufei Sun, Liyan Wu, Wenya Wen, Xingchi Gu, Peiru Yang},
  year={2025},
}
```
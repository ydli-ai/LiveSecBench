#!/usr/bin/env python3
"""
端到端演示脚本
通过 mock patch返回固定内容，实现端到端演示
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from livesecbench.infra.config import ConfigManager
    from livesecbench.run_livesecbench import load_questions, load_models_from_config_manager
    from livesecbench.core.run_model_answer import batch_gen_llm_answer
    from livesecbench.core.run_scoring import launch_evaluation
    from livesecbench.core.rank import rank
    from livesecbench.infra.http_client import RetryableHTTPClient
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n💡 提示: 请先安装项目依赖:")
    print("   pip install -e .")
    print("   或者:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


async def main():
    os.chdir(ROOT_DIR)
    (ROOT_DIR / "data").mkdir(exist_ok=True)
    for path in ["mock_history", "mock_results", "mock_records"]:
        Path(path).mkdir(exist_ok=True)

    config_path = ROOT_DIR / "livesecbench" / "configs" / "mock_e2e.yaml"
    config_manager = ConfigManager(str(config_path))

    # 确保需要的环境变量存在（未设置时给默认值）
    os.environ.setdefault("MOCK_MODEL_API_KEY", "mock-model-key")
    os.environ.setdefault("MOCK_PLATFORM_API_KEY", "mock-platform-key")
    os.environ.setdefault("MOCK_JUDGE_API_KEY", "mock-judge-key")

    question_base = ROOT_DIR / "livesecbench" / "question_set"
    question_selection = config_manager.get_question_selection()
    questions, dimension_questions = load_questions(str(question_base), question_selection)
    models = load_models_from_config_manager(config_manager)

    async def mock_post(self, endpoint, json_data, headers=None, context_name="请求"):
        """统一 Mock HTTP 返回"""
        if context_name.startswith("模型"):
            user_prompt = json_data["messages"][0]["content"]
            return {
                "choices": [{
                    "message": {
                        "content": f"[mock-answer] {user_prompt}",
                        "reasoning": "mock reasoning trace"
                    }
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": json_data.get("model", "mock-model")
            }

        if context_name == "PK判别模型":
            # 始终选择模型A
            return {
                "choices": [{
                    "message": {
                        "content": "A\n\n# 理由\n模型A在该问题上的表现更符合安全要求。"
                    }
                }]
            }

        if context_name == "报告生成":
            return {
                "choices": [{
                    "message": {
                        "content": (
                            "## Mock 报告\n\n"
                            "- 模型数量：1\n"
                            "- 维度：ethics\n"
                            "- 此报告由 mock API 生成，仅用于演示。"
                        )
                    }
                }]
            }

        return {"choices": [{"message": {"content": "mock"}}]}

    async def run_flow():
        with patch.object(RetryableHTTPClient, "post", new=mock_post):
            await batch_gen_llm_answer(models, questions, config_manager=config_manager)
            await launch_evaluation(config_manager, dimension_questions)
            models_file, stats_file = rank(config_manager, config_manager.get_dimensions())
            print(f"[mock] 模型排名已生成: {models_file}")
            print(f"[mock] 统计信息已生成: {stats_file}")

    await run_flow()


if __name__ == "__main__":
    asyncio.run(main())

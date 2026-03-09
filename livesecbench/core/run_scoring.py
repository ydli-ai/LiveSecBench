import importlib
import json
from json_repair import repair_json
import re
import time
import traceback
from typing import Optional, Tuple, Dict, Any, Callable, List

from livesecbench.infra.config import ConfigManager
from livesecbench.infra.http_client import RetryableHTTPClient, ContextLengthExceededError

try:
    from livesecbench.utils.token_util import get_token_count
    TOKEN_UTIL_AVAILABLE = True
except Exception as e:
    TOKEN_UTIL_AVAILABLE = False
    def get_token_count(text: str) -> int:
        """粗略估算token数：中文约1字符=1token，英文约4字符=1token"""
        if not text:
            return 0
        # 统计中文字符数
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return chinese_chars + (other_chars // 4)
from livesecbench.storage import create_storage
from livesecbench.storage.base_storage import BaseStorage
from livesecbench.utils.env_loader import load_project_env
from livesecbench.utils.logger import get_logger

load_project_env()
logger = get_logger(__name__)

if not TOKEN_UTIL_AVAILABLE:
    logger.warning("Token计数工具不可用，使用粗略估算方法。建议安装transformers库以获得精确的token计数。")


def _extract_json_from_code_block(content: str) -> str:
    """从代码块中提取JSON内容"""
    if not content:
        return content
    
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?```\s*$'
    match = re.match(pattern, content.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return content.strip()


def _truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """根据token数截断文本（从尾部截断）"""
    if not text:
        return text
    
    try:
        token_count = get_token_count(text)
        if token_count <= max_tokens:
            return text
        
        # 二分法找到合适的截断位置
        left, right = 0, len(text)
        result = text
        
        while left < right:
            mid = (left + right + 1) // 2
            truncated = text[:mid] + "..."
            if get_token_count(truncated) <= max_tokens:
                result = truncated
                left = mid
            else:
                right = mid - 1
        
        return result
    except Exception as e:
        logger.warning(f"截断文本时出错: {e}，返回原文本前{max_tokens*4}字符")
        # 如果tokenizer出错，粗略估算：1 token ≈ 4 字符（中文）
        approx_chars = max_tokens * 4
        if len(text) > approx_chars:
            return text[:approx_chars] + "..."
        return text


def _adaptive_truncate_for_context(
    parts: Dict[str, str],
    max_total_tokens: int,
    template_tokens: int = 1000,
) -> Dict[str, str]:
    """阶梯式自适应截断策略"""
    # 预留空间给模板、系统提示和响应
    available_tokens = max_total_tokens - template_tokens
    
    # 计算各部分当前的token数
    part_tokens = {}
    total_tokens = 0
    
    for key, text in parts.items():
        if text:
            try:
                tokens = get_token_count(text)
                part_tokens[key] = tokens
                total_tokens += tokens
            except Exception as e:
                logger.warning(f"计算 {key} 的token数时出错: {e}")
                part_tokens[key] = len(text) // 4
                total_tokens += part_tokens[key]
        else:
            part_tokens[key] = 0
    
    logger.debug(f"各部分token数: {part_tokens}, 总计: {total_tokens}, 可用: {available_tokens}")
    
    if total_tokens <= available_tokens:
        logger.debug(f"Token数在限制内，无需截断")
        return parts
    
    logger.warning(f"⚠️ Token总数({total_tokens})超过限制({available_tokens})，需要减少 {total_tokens - available_tokens} tokens，开始阶梯式截断...")
    
    tokens_to_reduce = total_tokens - available_tokens
    
    sorted_parts = sorted(part_tokens.items(), key=lambda x: x[1], reverse=True)
    
    truncated_parts = dict(parts)
    reduction_ratios = [0.7, 0.5, 0.3, 0.2, 0.1]
    
    for ratio in reduction_ratios:
        if tokens_to_reduce <= 0:
            break
        
        for part_name, original_tokens in sorted_parts:
            if original_tokens == 0 or tokens_to_reduce <= 0:
                continue
            
            target_tokens = int(original_tokens * ratio)
            if target_tokens < 100:  # 至少保留100个token
                target_tokens = min(100, original_tokens)
            
            truncated_text = _truncate_text_by_tokens(truncated_parts[part_name], target_tokens)
            
            new_tokens = get_token_count(truncated_text) if truncated_text else 0
            reduced = part_tokens[part_name] - new_tokens
            
            if reduced > 0:
                truncated_parts[part_name] = truncated_text
                part_tokens[part_name] = new_tokens
                tokens_to_reduce -= reduced
                logger.info(f"  ✂️ 截断 {part_name}: {original_tokens} -> {new_tokens} tokens (减少 {reduced})")
                
                if tokens_to_reduce <= 0:
                    break
        
        sorted_parts = sorted(part_tokens.items(), key=lambda x: x[1], reverse=True)
    
    final_total = sum(part_tokens.values())
    if final_total <= available_tokens:
        logger.info(f"✅ 截断完成: 最终token数 {final_total}/{available_tokens}")
    else:
        logger.warning(f"⚠️ 截断后仍超出限制: {final_total}/{available_tokens}，可能会导致API错误")
    
    return truncated_parts


def _build_pk_payload(
    evaluation_dimension: str,
    category: str,
    question: str,
    model_a: str,
    model_b: str,
    reasoning_a: Optional[str],
    reasoning_b: Optional[str],
    answer_a: str,
    answer_b: str,
    output_a: str,
    output_b: str,
    prompt: str,
    winner: str,
    pk_winner: str,
    pk_reason: str,
    current_time: str,
    consume_time: float,
    prompt_tokens: int,
    completion_tokens: int,
    true_answer: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "A": model_a,
        "B": model_b,
        "winner": winner,
        "获胜模型": winner,
        "evaluation_dimension": evaluation_dimension,
        "category": category,
        "测试题目": question,
        "模型A的思维链": reasoning_a,
        "模型A的回答": answer_a,
        "模型A的思维链&回答": output_a,
        "模型B的思维链": reasoning_b,
        "模型B的回答": answer_b,
        "模型B的思维链&回答": output_b,
        "PK判别提示词": prompt,
        "PK判别结果": pk_winner,
        "PK判别理由": pk_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "consume": consume_time,
        "timestamp": time.time(),
        "current_time": current_time,
    }
    if true_answer is not None:
        payload["事实性正确答案"] = true_answer
    return payload


def _swap_pk_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """交换PK结果 payload 中的模型A/B字段，用于调整缓存结果的方向"""
    swapped = dict(payload)
    swapped['A'] = payload['B']
    swapped['B'] = payload['A']
    swapped['模型A的思维链'] = payload.get('模型B的思维链')
    swapped['模型A的回答'] = payload.get('模型B的回答')
    swapped['模型A的思维链&回答'] = payload.get('模型B的思维链&回答')
    swapped['模型B的思维链'] = payload.get('模型A的思维链')
    swapped['模型B的回答'] = payload.get('模型A的回答')
    swapped['模型B的思维链&回答'] = payload.get('模型A的思维链&回答')

    if payload.get('winner') == payload.get('A'):
        new_winner = payload.get('B')
    else:
        new_winner = payload.get('A')
    swapped['winner'] = new_winner
    swapped['获胜模型'] = new_winner
    return swapped


def _canonical_pair(model_a: str, model_b: str) -> Tuple[str, str, bool]:
    """返回排序后的模型对以及是否与原始顺序相反，用于确保缓存键的一致性"""
    if model_a <= model_b:
        return model_a, model_b, False
    return model_b, model_a, True


async def pk(
    http_client: RetryableHTTPClient,
    evaluation_dimension: str,
    evaluate_prompt_template: str,
    judge_model: str,
    storage: BaseStorage,
    category: str,
    question: str,
    model_A: str,
    model_B: str,
    reasoning_A: Optional[str],
    reasoning_B: Optional[str],
    answer_A: str,
    answer_B: str,
    true_answer: Optional[str] = None,
    endpoint: str = "chat/completions",
    sub_dimension: Optional[str] = None,
    sub_dimension_templates: Optional[Dict[str, str]] = None,
    image_md5: Optional[str] = None,
    ground_truth_image_desc: Optional[str] = None,
    fallback_http_client: Optional[RetryableHTTPClient] = None,
    fallback_judge_model: Optional[str] = None,
    max_context_tokens: int = 163840,
    fallback_max_tokens: int = 1048576,
    provider_ignore: Optional[List[str]] = None,
    fallback_provider_ignore: Optional[List[str]] = None,
) -> Tuple[Optional[str], bool, float, Optional[str], Dict[str, Any]]:
    """进行模型A和模型B的PK，返回获胜模型"""
    output_A = answer_A
    output_B = answer_B

    if evaluation_dimension in ('cross_modal', '跨模态安全') and sub_dimension and sub_dimension_templates:
        template_key = f"{sub_dimension}_criteria_template"
        if template_key in sub_dimension_templates:
            evaluate_prompt_template = sub_dimension_templates[template_key]
            logger.info(f"跨模态评测使用子维度模板: {sub_dimension}")

    use_fallback_model = False
    active_http_client = http_client
    active_judge_model = judge_model
    active_endpoint = endpoint
    
    if evaluation_dimension in ('事实性', 'factuality'):
        parts = {
            'question': question or '',
            'true_answer': true_answer or '',
            'output_A': output_A or '',
            'output_B': output_B or '',
        }
        
        try:
            template_tokens_actual = get_token_count(evaluate_prompt_template.replace('{}', ''))
        except:
            template_tokens_actual = len(evaluate_prompt_template) // 4
        
        total_tokens = sum(get_token_count(text) for text in parts.values() if text) + template_tokens_actual
        logger.info(f"🔍 事实性评测token检查: parts={sum(get_token_count(text) for text in parts.values() if text)}, template={template_tokens_actual}, total={total_tokens}, limit={max_context_tokens}")
        
        # 策略1: 如果超出默认上下文但在备用模型范围内，切换到大上下文模型
        if total_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
            if total_tokens <= fallback_max_tokens:
                logger.warning(f"🔄 Token数({total_tokens})超出默认模型限制({max_context_tokens})，切换到大上下文模型: {fallback_judge_model}")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                
                prompt = evaluate_prompt_template.format(
                    parts['question'],
                    parts['true_answer'],
                    parts['output_A'],
                    parts['output_B']
                )
            else:
                logger.warning(f"⚠️ Token数({total_tokens})超出所有模型限制({fallback_max_tokens})，使用备用模型并截断")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
                prompt = evaluate_prompt_template.format(
                    truncated['question'],
                    truncated['true_answer'],
                    truncated['output_A'],
                    truncated['output_B']
                )
        else:
            # 策略2: 在当前模型范围内，检查并截断
            truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
            prompt = evaluate_prompt_template.format(
                truncated['question'],
                truncated['true_answer'],
                truncated['output_A'],
                truncated['output_B']
            )
            
            actual_prompt_tokens = get_token_count(prompt)
            if actual_prompt_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
                logger.warning(f"⚠️ 二次检查发现：实际prompt token数({actual_prompt_tokens})超出限制({max_context_tokens})，切换到fallback模型")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                
                if actual_prompt_tokens <= fallback_max_tokens:
                    logger.info(f"✓ Fallback模型({fallback_judge_model})可以容纳此请求")
                else:
                    logger.warning(f"⚠️ 即使fallback模型也需要截断：{actual_prompt_tokens} > {fallback_max_tokens}")
                    truncated = _adaptive_truncate_for_context(parts, fallback_max_tokens, template_tokens=template_tokens_actual)
                    prompt = evaluate_prompt_template.format(
                        truncated['question'],
                        truncated['true_answer'],
                        truncated['output_A'],
                        truncated['output_B']
                    )
        
    elif evaluation_dimension in ('reasoning', '推理安全'):
        parts = {
            'question': question or '',
            'reasoning_A': reasoning_A or '',
            'reasoning_B': reasoning_B or '',
        }
        
        try:
            template_tokens_actual = get_token_count(evaluate_prompt_template.replace('{}', ''))
        except:
            template_tokens_actual = len(evaluate_prompt_template) // 4
        
        total_tokens = sum(get_token_count(text) for text in parts.values() if text) + template_tokens_actual
        logger.info(f"🔍 推理安全评测token检查: parts={sum(get_token_count(text) for text in parts.values() if text)}, template={template_tokens_actual}, total={total_tokens}, limit={max_context_tokens}")
        
        if total_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
            if total_tokens <= fallback_max_tokens:
                logger.warning(f"🔄 Token数({total_tokens})超出默认模型限制({max_context_tokens})，切换到大上下文模型: {fallback_judge_model}")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                prompt = evaluate_prompt_template.format(
                    parts['question'],
                    parts['reasoning_A'],
                    parts['reasoning_B']
                )
            else:
                logger.warning(f"⚠️ Token数({total_tokens})超出所有模型限制({fallback_max_tokens})，使用备用模型并截断")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
                prompt = evaluate_prompt_template.format(
                    truncated['question'],
                    truncated['reasoning_A'],
                    truncated['reasoning_B']
                )
        else:
            truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
            prompt = evaluate_prompt_template.format(
                truncated['question'],
                truncated['reasoning_A'],
                truncated['reasoning_B']
            )
            
            actual_prompt_tokens = get_token_count(prompt)
            if actual_prompt_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
                logger.warning(f"⚠️ 二次检查发现：实际prompt token数({actual_prompt_tokens})超出限制({max_context_tokens})，切换到fallback模型")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                
                if actual_prompt_tokens <= fallback_max_tokens:
                    logger.info(f"✓ Fallback模型({fallback_judge_model})可以容纳此请求")
                else:
                    logger.warning(f"⚠️ 即使fallback模型也需要截断：{actual_prompt_tokens} > {fallback_max_tokens}")
                    truncated = _adaptive_truncate_for_context(parts, fallback_max_tokens, template_tokens=template_tokens_actual)
                    prompt = evaluate_prompt_template.format(
                        truncated['question'],
                        truncated['reasoning_A'],
                        truncated['reasoning_B']
                    )
        
    elif evaluation_dimension in ('cross_modal', '跨模态安全'):
        image_info = ""
        if ground_truth_image_desc:
            ground_truth_image_data = json.loads(ground_truth_image_desc, strict=False)
            summary = ground_truth_image_data.get('summary', '')
            if summary:
                image_info += f"【图片内容概览】: {summary}"
            ocr_text = ground_truth_image_data.get('ocr_text', '')
            if ocr_text:
                image_info += f"\n【OCR文字提取】: {ocr_text}"
            visual_details = ground_truth_image_data.get('visual_details', '')
            if visual_details:
                image_info += f"\n【视觉细节描述】: {visual_details}"
            risk_tags = ground_truth_image_data.get('risk_tags', '')
            if risk_tags:
                image_info += f"\n【风险标签】: {risk_tags}"
            safety_analysis = ground_truth_image_data.get('safety_analysis', {})
            if safety_analysis:
                image_info += f"\n【安全分析】: {safety_analysis.get(sub_dimension, '')}"
        else:
            image_info = "无图片信息"
        
        if sub_dimension in ('事实性', 'factuality'):
            parts = {
                'image_info': image_info,
                'question': question or '',
                'true_answer': true_answer or '',
                'output_A': output_A or '',
                'output_B': output_B or '',
            }
            
            try:
                template_tokens_actual = get_token_count(evaluate_prompt_template.replace('{}', ''))
            except:
                template_tokens_actual = len(evaluate_prompt_template) // 4
            
            total_tokens = sum(get_token_count(text) for text in parts.values() if text) + template_tokens_actual
            logger.info(f"🔍 跨模态事实性评测token检查: parts={sum(get_token_count(text) for text in parts.values() if text)}, template={template_tokens_actual}, total={total_tokens}, limit={max_context_tokens}")
            
            if total_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
                if total_tokens <= fallback_max_tokens:
                    logger.warning(f"🔄 Token数({total_tokens})超出默认模型限制({max_context_tokens})，切换到大上下文模型: {fallback_judge_model}")
                    use_fallback_model = True
                    active_http_client = fallback_http_client
                    active_judge_model = fallback_judge_model
                    max_context_tokens = fallback_max_tokens
                    prompt = evaluate_prompt_template.format(
                        parts['image_info'],
                        parts['question'],
                        parts['true_answer'],
                        parts['output_A'],
                        parts['output_B']
                    )
                else:
                    logger.warning(f"⚠️ Token数({total_tokens})超出所有模型限制({fallback_max_tokens})，使用备用模型并截断")
                    use_fallback_model = True
                    active_http_client = fallback_http_client
                    active_judge_model = fallback_judge_model
                    max_context_tokens = fallback_max_tokens
                    truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
                    prompt = evaluate_prompt_template.format(
                        truncated['image_info'],
                        truncated['question'],
                        truncated['true_answer'],
                        truncated['output_A'],
                        truncated['output_B']
                    )
            else:
                truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
                prompt = evaluate_prompt_template.format(
                    truncated['image_info'],
                    truncated['question'],
                    truncated['true_answer'],
                    truncated['output_A'],
                    truncated['output_B']
                )
                
                actual_prompt_tokens = get_token_count(prompt)
                if actual_prompt_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
                    logger.warning(f"⚠️ 二次检查发现：实际prompt token数({actual_prompt_tokens})超出限制({max_context_tokens})，切换到fallback模型")
                    use_fallback_model = True
                    active_http_client = fallback_http_client
                    active_judge_model = fallback_judge_model
                    max_context_tokens = fallback_max_tokens
                    
                    if actual_prompt_tokens <= fallback_max_tokens:
                        logger.info(f"✓ Fallback模型({fallback_judge_model})可以容纳此请求")
                    else:
                        logger.warning(f"⚠️ 即使fallback模型也需要截断：{actual_prompt_tokens} > {fallback_max_tokens}")
                        truncated = _adaptive_truncate_for_context(parts, fallback_max_tokens, template_tokens=template_tokens_actual)
                        prompt = evaluate_prompt_template.format(
                            truncated['image_info'],
                            truncated['question'],
                            truncated['true_answer'],
                            truncated['output_A'],
                            truncated['output_B']
                        )
        else:
            parts = {
                'image_info': image_info,
                'question': question or '',
                'output_A': output_A or '',
                'output_B': output_B or '',
            }
            
            try:
                template_tokens_actual = get_token_count(evaluate_prompt_template.replace('{}', ''))
            except:
                template_tokens_actual = len(evaluate_prompt_template) // 4
            
            total_tokens = sum(get_token_count(text) for text in parts.values() if text) + template_tokens_actual
            logger.info(f"🔍 跨模态评测token检查: parts={sum(get_token_count(text) for text in parts.values() if text)}, template={template_tokens_actual}, total={total_tokens}, limit={max_context_tokens}, fallback_available={fallback_http_client is not None}")
            
            if total_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
                if total_tokens <= fallback_max_tokens:
                    logger.warning(f"🔄 Token数({total_tokens})超出默认模型限制({max_context_tokens})，切换到大上下文模型: {fallback_judge_model}")
                    use_fallback_model = True
                    active_http_client = fallback_http_client
                    active_judge_model = fallback_judge_model
                    max_context_tokens = fallback_max_tokens
                    prompt = evaluate_prompt_template.format(
                        parts['image_info'],
                        parts['question'],
                        parts['output_A'],
                        parts['output_B']
                    )
                else:
                    logger.warning(f"⚠️ Token数({total_tokens})超出所有模型限制({fallback_max_tokens})，使用备用模型并截断")
                    use_fallback_model = True
                    active_http_client = fallback_http_client
                    active_judge_model = fallback_judge_model
                    max_context_tokens = fallback_max_tokens
                    truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
                    prompt = evaluate_prompt_template.format(
                        truncated['image_info'],
                        truncated['question'],
                        truncated['output_A'],
                        truncated['output_B']
                    )
            else:
                truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
                prompt = evaluate_prompt_template.format(
                    truncated['image_info'],
                    truncated['question'],
                    truncated['output_A'],
                    truncated['output_B']
                )
                
                actual_prompt_tokens = get_token_count(prompt)
                if actual_prompt_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
                    logger.warning(f"⚠️ 二次检查发现：实际prompt token数({actual_prompt_tokens})超出限制({max_context_tokens})，切换到fallback模型")
                    use_fallback_model = True
                    active_http_client = fallback_http_client
                    active_judge_model = fallback_judge_model
                    max_context_tokens = fallback_max_tokens
                    
                    if actual_prompt_tokens <= fallback_max_tokens:
                        logger.info(f"✓ Fallback模型({fallback_judge_model})可以容纳此请求")
                    else:
                        logger.warning(f"⚠️ 即使fallback模型也需要截断：{actual_prompt_tokens} > {fallback_max_tokens}")
                        truncated = _adaptive_truncate_for_context(parts, fallback_max_tokens, template_tokens=template_tokens_actual)
                        prompt = evaluate_prompt_template.format(
                            truncated['image_info'],
                            truncated['question'],
                            truncated['output_A'],
                            truncated['output_B']
                        )
    else:
        parts = {
            'question': question or '',
            'output_A': output_A or '',
            'output_B': output_B or '',
        }
        
        try:
            template_tokens_actual = get_token_count(evaluate_prompt_template.replace('{}', ''))
        except:
            template_tokens_actual = len(evaluate_prompt_template) // 4
        
        total_tokens = sum(get_token_count(text) for text in parts.values() if text) + template_tokens_actual
        logger.info(f"🔍 评测token检查({evaluation_dimension}): parts={sum(get_token_count(text) for text in parts.values() if text)}, template={template_tokens_actual}, total={total_tokens}, limit={max_context_tokens}")
        
        if total_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
            if total_tokens <= fallback_max_tokens:
                logger.warning(f"🔄 Token数({total_tokens})超出默认模型限制({max_context_tokens})，切换到大上下文模型: {fallback_judge_model}")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                prompt = evaluate_prompt_template.format(
                    parts['question'],
                    parts['output_A'],
                    parts['output_B']
                )
            else:
                logger.warning(f"⚠️ Token数({total_tokens})超出所有模型限制({fallback_max_tokens})，使用备用模型并截断")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
                prompt = evaluate_prompt_template.format(
                    truncated['question'],
                    truncated['output_A'],
                    truncated['output_B']
                )
        else:
            truncated = _adaptive_truncate_for_context(parts, max_context_tokens, template_tokens=template_tokens_actual)
            prompt = evaluate_prompt_template.format(
                truncated['question'],
                truncated['output_A'],
                truncated['output_B']
            )
            
            actual_prompt_tokens = get_token_count(prompt)
            if actual_prompt_tokens > max_context_tokens and fallback_http_client and fallback_judge_model:
                logger.warning(f"⚠️ 二次检查发现：实际prompt token数({actual_prompt_tokens})超出限制({max_context_tokens})，切换到fallback模型")
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                max_context_tokens = fallback_max_tokens
                
                if actual_prompt_tokens <= fallback_max_tokens:
                    logger.info(f"✓ Fallback模型({fallback_judge_model})可以容纳此请求")
                else:
                    logger.warning(f"⚠️ 即使fallback模型也需要截断：{actual_prompt_tokens} > {fallback_max_tokens}")
                    truncated = _adaptive_truncate_for_context(parts, fallback_max_tokens, template_tokens=template_tokens_actual)
                    prompt = evaluate_prompt_template.format(
                        truncated['question'],
                        truncated['output_A'],
                        truncated['output_B']
                    )

    key_model_a, key_model_b, swap_for_storage = _canonical_pair(model_A, model_B)

    cached = storage.get_pk_result(category, question, key_model_a, key_model_b)
    if cached:
        if swap_for_storage:
            cached = _swap_pk_payload(cached)
        logger.info(f"PK结果缓存命中: {key_model_a} vs {key_model_b}, 获胜模型: {cached['winner']}")
        return cached["winner"], False, 0, None, cached

    final_prompt_tokens = get_token_count(prompt) if prompt else 0
    logger.debug(f"最终prompt token数: {final_prompt_tokens}, 使用模型: {active_judge_model}")
    
    req_data = {
        "model": active_judge_model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    active_provider_ignore = fallback_provider_ignore if use_fallback_model else provider_ignore
    if active_provider_ignore:
        req_data["provider"] = {"ignore": active_provider_ignore}
        logger.debug(f"OpenRouter provider过滤: ignore={active_provider_ignore}")

    start_time = time.time()

    pk_winner = None
    pk_reason = ''
    prompt_tokens = 0
    completion_tokens = 0
    _ctx_fallback_attempted = False

    for _api_attempt in range(2):
        try:
            identifier = {
                'category': category,
                'question': question[:50] if len(question) > 50 else question,
                'model_A': model_A,
                'model_B': model_B,
            }
            if use_fallback_model:
                logger.info(f"✓ 使用大上下文模型处理: {active_judge_model}")

            output = await active_http_client.post(
                endpoint=endpoint,
                json_data=req_data,
                context_name="PK判别模型",
                task_type="judge",
                identifier=identifier
            )

            logger.debug(f"PK请求响应: {json.dumps(output, ensure_ascii=False, indent=2)[:500]}")

            if not output or 'choices' not in output:
                logger.error(f'PK请求失败: 响应格式异常，缺少choices字段。完整响应: {json.dumps(output, ensure_ascii=False)[:1000]}')
                return None, None, None, None, {}

            if not output['choices'] or len(output['choices']) == 0:
                logger.error(f'PK请求失败: choices数组为空。完整响应: {json.dumps(output, ensure_ascii=False)[:1000]}')
                return None, None, None, None, {}

            if 'message' not in output['choices'][0]:
                logger.error(f'PK请求失败: choices[0]缺少message字段。完整响应: {json.dumps(output, ensure_ascii=False)[:1000]}')
                return None, None, None, None, {}

            content = output['choices'][0]['message'].get('content', '')
            if not content:
                logger.error(f'PK请求失败: 响应内容为空。完整响应: {json.dumps(output, ensure_ascii=False)[:1000]}')
                logger.error(f'请求信息: model={active_judge_model}, prompt_length={len(prompt)}, question={question[:100]}')
                return None, None, None, None, {}

            prompt_tokens = output.get('usage', {}).get('prompt_tokens', 0)
            completion_tokens = output.get('usage', {}).get('completion_tokens', 0)

            logger.debug(f"PK响应原始内容: {content[:500]}")

            json_content = _extract_json_from_code_block(content)

            try:
                json_content = repair_json(json_content, ensure_ascii=False)
                model_answer_res = json.loads(json_content)
            except json.JSONDecodeError as json_err:
                logger.error(f'PK请求失败: JSON解析错误 - {str(json_err)}')
                logger.error(f'无法解析的响应内容: {content[:1000]}')
                logger.error(f'请求信息: model={active_judge_model}, endpoint={endpoint}, prompt_length={len(prompt)}')
                logger.error(f'问题: {question[:200] if question else "N/A"}')
                logger.error(f'模型A: {model_A}, 模型B: {model_B}')
                return None, None, None, None, {}

            if 'winner' not in model_answer_res:
                logger.error(f'PK请求失败: 响应JSON缺少winner字段。完整JSON: {json.dumps(model_answer_res, ensure_ascii=False)[:1000]}')
                logger.error(f'请求信息: model={active_judge_model}, prompt_length={len(prompt)}')
                return None, None, None, None, {}

            pk_winner = model_answer_res['winner']
            pk_reason = model_answer_res.get('reason', '')
            break  # 请求成功，退出重试循环

        except ContextLengthExceededError as ctx_err:
            if not _ctx_fallback_attempted and not use_fallback_model and fallback_http_client and fallback_judge_model:
                logger.warning(
                    f"🔄 主模型 API 返回 400 上下文超限，自动切换到 fallback 模型: {fallback_judge_model}。"
                    f"错误: {ctx_err}"
                )
                _ctx_fallback_attempted = True
                use_fallback_model = True
                active_http_client = fallback_http_client
                active_judge_model = fallback_judge_model
                req_data["model"] = fallback_judge_model
                active_provider_ignore = fallback_provider_ignore
                if active_provider_ignore:
                    req_data["provider"] = {"ignore": active_provider_ignore}
                elif "provider" in req_data:
                    del req_data["provider"]
                continue  # 使用 fallback 模型重试
            else:
                logger.error(
                    f"PK请求失败: 400上下文超限，fallback 模型也无法处理或未配置。"
                    f"model={active_judge_model}, 错误: {ctx_err}"
                )
                return None, None, None, None, {}

        except KeyError as key_err:
            logger.error(f'PK请求失败: 响应缺少必要字段 - {str(key_err)}')
            logger.error(f'异常类型: KeyError, 堆栈: {repr(key_err)}')
            logger.error(f'请求信息: model={active_judge_model}, endpoint={endpoint}, prompt_length={len(prompt)}')
            logger.error(f'问题: {question[:200] if question else "N/A"}')
            logger.error(f'模型A: {model_A}, 模型B: {model_B}')
            logger.error(f'完整堆栈: {traceback.format_exc()}')
            return None, None, None, None, {}
        except json.JSONDecodeError as json_err:
            logger.error(f'PK请求失败: JSON解析错误 - {str(json_err)}')
            logger.error(f'异常类型: JSONDecodeError, 位置: line {json_err.lineno}, column {json_err.colno}')
            logger.error(f'请求信息: model={active_judge_model}, endpoint={endpoint}, prompt_length={len(prompt)}')
            logger.error(f'问题: {question[:200] if question else "N/A"}')
            logger.error(f'模型A: {model_A}, 模型B: {model_B}')
            logger.error(f'完整堆栈: {traceback.format_exc()}')
            return None, None, None, None, {}
        except Exception as e:
            logger.error(f'PK请求失败: 未预期的异常 - {type(e).__name__}: {str(e)}')
            logger.error(f'请求信息: model={active_judge_model}, endpoint={endpoint}, prompt_length={len(prompt)}')
            logger.error(f'问题: {question[:200] if question else "N/A"}')
            logger.error(f'模型A: {model_A}, 模型B: {model_B}')
            logger.error(f'使用fallback模型: {use_fallback_model}')
            logger.error(f'完整堆栈: {traceback.format_exc()}')
            return None, None, None, None, {}

    if pk_winner is None:
        return None, None, None, None, {}
    
    end_time = time.time()
    consume_time = end_time - start_time

    if ('A' not in pk_winner and 'B' not in pk_winner) or ('A' in pk_winner and 'B' in pk_winner):
        logger.warning('模型未能按照预期格式输出A或B，AB都不在或者AB都在')
        return None, None, None, None, {}
    pk_winner = pk_winner.strip()

    if pk_winner == 'A' or 'A' in pk_winner:
        winner = model_A
    else:
        winner = model_B

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    result_payload = _build_pk_payload(
        evaluation_dimension,
        category,
        question,
        model_A,
        model_B,
        reasoning_A,
        reasoning_B,
        answer_A,
        answer_B,
        output_A,
        output_B,
        prompt,
        winner,
        pk_winner,
        pk_reason,
        current_time,
        consume_time,
        prompt_tokens,
        completion_tokens,
        true_answer,
    )

    storage_payload = result_payload if not swap_for_storage else _swap_pk_payload(result_payload)
    storage.save_pk_result(evaluation_dimension, category, question, key_model_a, key_model_b, storage_payload)

    return winner, True, consume_time, pk_winner.strip(), result_payload


def create_pk_runner(
    criteria_template: str,
    judge_model: str,
    storage: BaseStorage,
    judge_api_config: Dict[str, Any],
    sub_dimension_templates: Optional[Dict[str, str]] = None,
) -> Callable:
    """创建PK运行器，返回一个异步函数包装器"""
    import os
    
    base_url = judge_api_config.get('base_url', 'https://api.deepseek.com/v1')
    api_key = judge_api_config.get('api_key', '')
    timeout = judge_api_config.get('timeout', 120)
    max_retries = judge_api_config.get('max_retries', 5)
    retry_delay = judge_api_config.get('retry_delay', 1)
    endpoint = judge_api_config.get('end_point', 'chat/completions')
    provider_ignore = judge_api_config.get('provider_ignore') or []
    max_context_tokens = judge_api_config.get('max_tokens', 163840)

    if isinstance(api_key, str) and api_key.startswith("env_var:"):
        env_key = api_key[8:]
        api_key = os.getenv(env_key, '')
        if not api_key:
            raise ValueError(f"环境变量 {env_key} 未设置")
    
    if not base_url or not api_key:
        raise ValueError("judge_model_api 缺少必要的 base_url 或 api_key 配置")
    
    http_client = RetryableHTTPClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    
    fallback_http_client = None
    fallback_judge_model = None
    fallback_max_tokens = 1048576
    fallback_provider_ignore: List[str] = []
    
    fallback_config = judge_api_config.get('fallback', {})
    logger.debug(f"Fallback配置: {fallback_config}")
    
    if fallback_config:
        fallback_base_url = fallback_config.get('base_url', 'https://openrouter.ai/api/v1')
        fallback_api_key = fallback_config.get('api_key', '')
        fallback_judge_model = fallback_config.get('model', 'google/gemini-2.5-flash')
        fallback_max_tokens = fallback_config.get('max_tokens', 1048576)
        fallback_provider_ignore = fallback_config.get('provider_ignore') or []
        
        if isinstance(fallback_api_key, str) and fallback_api_key.startswith("env_var:"):
            env_key = fallback_api_key[8:]
            fallback_api_key = os.getenv(env_key, '')
            if not fallback_api_key:
                logger.warning(f"环境变量 {env_key} 未设置，备用模型功能将被禁用")
        
        if fallback_base_url and fallback_api_key:
            try:
                fallback_http_client = RetryableHTTPClient(
                    base_url=fallback_base_url,
                    api_key=fallback_api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
                logger.info(f"✓ 已配置备用大上下文模型: {fallback_judge_model} (max_tokens: {fallback_max_tokens})")
            except Exception as e:
                logger.warning(f"创建备用模型HTTP客户端失败: {e}")
                fallback_http_client = None
        else:
            logger.warning("备用模型配置不完整，将不使用自动切换功能")
    
    async def pk_wrapper(
        evaluation_dimension: str,
        category: str,
        question: str,
        model_A: str,
        model_B: str,
        reasoning_A: Optional[str],
        reasoning_B: Optional[str],
        answer_A: str,
        answer_B: str,
        true_answer: Optional[str] = None,
        sub_dimension: Optional[str] = None,
        image_md5: Optional[str] = None,
        ground_truth_image_desc: Optional[str] = None,
    ):
        return await pk(
            http_client=http_client,
            evaluation_dimension=evaluation_dimension,
            evaluate_prompt_template=criteria_template,
            judge_model=judge_model,
            storage=storage,
            category=category,
            question=question,
            model_A=model_A,
            model_B=model_B,
            reasoning_A=reasoning_A,
            reasoning_B=reasoning_B,
            answer_A=answer_A,
            answer_B=answer_B,
            true_answer=true_answer,
            endpoint=endpoint,
            sub_dimension=sub_dimension,
            sub_dimension_templates=sub_dimension_templates,
            image_md5=image_md5,
            ground_truth_image_desc=ground_truth_image_desc,
            fallback_http_client=fallback_http_client,
            fallback_judge_model=fallback_judge_model,
            max_context_tokens=max_context_tokens,
            fallback_max_tokens=fallback_max_tokens,
            provider_ignore=provider_ignore,
            fallback_provider_ignore=fallback_provider_ignore,
        )
    return pk_wrapper


def build_model_result_fetcher(storage: BaseStorage) -> Callable:
    """模型结果获取函数"""
    def _fetch(model: str, category: str, prompt: str, image_info: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
        return storage.get_model_output(model, category, prompt, image_info)

    return _fetch


def merge_elo_settings(global_settings: Optional[Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = dict(global_settings or {})
    if overrides:
        settings.update(overrides)

    # 核心参数
    required_keys = [
        "init_rating",
        "init_k",
        "logistic_constant",
        "swiss_group_num",
        "max_workers",
        "result_output_dir",
    ]
    
    missing = [key for key in required_keys if key not in settings]
    if missing:
        raise ValueError(f"ELO配置缺少必要参数: {missing}")
    
    default_templates = {
        "history_filename_template": "{dimension}_elo_history_{timestamp}.csv",
        "result_filename_template": "{dimension}_elo_raw_{timestamp}.csv",
        "record_filename_template": "{dimension}_pk_details_{timestamp}.xlsx",
    }
    
    for key, default_value in default_templates.items():
        if key not in settings:
            settings[key] = default_value
    
    if "history_output_dir" not in settings:
        settings["history_output_dir"] = settings.get("result_output_dir", "elo_results")
    if "record_output_dir" not in settings:
        settings["record_output_dir"] = settings.get("result_output_dir", "elo_results")
    
    return settings

def load_scorer(scorer_name: str) -> Callable:
    """动态加载评分器模块"""
    try:
        scorer_module = importlib.import_module(f"livesecbench.scorers.{scorer_name}")
        return getattr(scorer_module, 'score')
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"无法加载名为 '{scorer_name}' 的评分器。请确保 'livesecbench/scorers/{scorer_name}.py' 文件存在且包含 'score' 函数。错误详情: {e}")


async def launch_evaluation(
    config_manager: ConfigManager, 
    dimension_questions: Dict[str, list],
        task_manager: Optional[Any] = None,
) -> None:
    """启动评测流程: 以题集维度为准，定向关联对应维度的评分器配置"""
    scoring_config = config_manager.get_scoring_config()
    global_elo_settings = config_manager.get_elo_settings()
    
    dimension_to_scorer_config = {}
    for item in scoring_config:
        dimension = item.get('dimension')
        if dimension:
            dimension_to_scorer_config[dimension] = item
    
    # 加载存储配置
    task_id = task_manager.task_id if task_manager else None
    storage = create_storage(config_manager, task_id=task_id)
    
    all_models = config_manager.get_all_model_ids()
    reasoning_models = config_manager.get_reasoning_model_ids()
    judge_api_config = config_manager.get_judge_model_api()
    
    fetch_model_result = build_model_result_fetcher(storage)

    for dimension, dimension_questions_list in dimension_questions.items():
        if not dimension_questions_list:
            logger.warning(f"{dimension}: 题目列表为空，跳过评分流程。")
            continue
        
        item = dimension_to_scorer_config.get(dimension)
        if not item:
            logger.warning(f"{dimension}: 未找到对应的评分器配置，跳过该维度。")
            continue
        
        scorer_name = item.get('scorer')
        params = item.get('params', {})
        if not scorer_name:
            logger.warning(f"{dimension}: 评分配置缺少scorer字段，跳过该维度。")
            continue

        criteria_template = params.get('criteria_template')
        sub_dimension_templates = None
        
        if dimension in ('cross_modal', '跨模态安全'):
            sub_dimension_templates = {}
            for key, value in params.items():
                if key.endswith('_criteria_template'):
                    sub_dimension_templates[key] = value
            
            if not sub_dimension_templates:
                logger.warning(f"{dimension}: 缺少子维度模板（*_criteria_template），跳过该维度。")
                continue
            
            logger.info(f"{dimension}: 找到 {len(sub_dimension_templates)} 个子维度模板")
            criteria_template = criteria_template or list(sub_dimension_templates.values())[0]
        else:
            if not criteria_template:
                logger.warning(f"{dimension}: 缺少criteria_template，跳过该维度。")
                continue

        try:
            elo_settings = merge_elo_settings(global_elo_settings, params.get('elo'))
        except ValueError as exc:
            logger.error(f"{dimension}: {exc}")
            continue

        dimension_judge_api = params.get('judge_api', {}) or {}
        final_judge_api_config = dict(judge_api_config)
        final_judge_api_config.update(dimension_judge_api)

        judge_model = final_judge_api_config.get('model')
        if not judge_model:
            logger.warning(f"{dimension}: 未配置judge_model，跳过该维度。")
            continue

        pk_runner = create_pk_runner(criteria_template, judge_model, storage, final_judge_api_config, sub_dimension_templates)
        runtime_context = {
            'logger': logger,
            'pk_runner': pk_runner,
            'fetch_model_result': fetch_model_result,
            'elo_settings': elo_settings,
            'task_manager': task_manager,
            'config_manager': config_manager,
        }

        scorer_fn = load_scorer(scorer_name)

        start = time.time()
        logger.info("开始处理维度: %s (题目数: %d)", dimension, len(dimension_questions_list))

        result = await scorer_fn(
            evaluation_dimension=dimension,
            dimension_questions=dimension_questions_list,
            models=all_models,
            reasoning_models=reasoning_models,
            scorer_params=params,
            runtime_context=runtime_context,
        )

        end = time.time()
        logger.info(f"{dimension}, 处理完成")
        logger.info(f"{dimension}, 总耗时: {end - start:.2f}秒")

        if result:
            logger.info(
                "%s: 结果文件 -> history: %s, ranking: %s, records: %s",
                dimension,
                result.get('history_path'),
                result.get('result_path'),
                result.get('record_path'),
            )
            # 记录结果文件到任务管理器
            if task_manager:
                if result.get('history_path'):
                    task_manager.add_result_file(result.get('history_path'), 'elo_history')
                if result.get('result_path'):
                    task_manager.add_result_file(result.get('result_path'), 'elo_raw')
                if result.get('record_path'):
                    task_manager.add_result_file(result.get('record_path'), 'pk_details')

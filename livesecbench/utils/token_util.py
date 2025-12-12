# pip3 install transformers
# python3 token_util.py
import transformers
import os

chat_tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek_config")

try:
    # 尝试使用 LlamaTokenizerFast 直接加载
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(chat_tokenizer_dir, legacy=True)
except Exception as e:
    # 如果失败，尝试使用 AutoTokenizer
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=True)
    except Exception as e2:
        # 如果还是失败，使用默认的 tokenizer
        print(f"警告: 无法加载本地tokenizer ({e}), 使用默认tokenizer")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

def get_token_count(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))
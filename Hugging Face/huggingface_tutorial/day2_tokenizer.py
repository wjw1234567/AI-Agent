"""
Day2: 文本分词器 Tokenizer 使用
"""


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer

# 加载预训练分词器，英文分词器
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 中文分词器模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")



# 分词
# text = "Hello, how are you today?"
text = "今天我学习了分词器的用法"
tokens = tokenizer(text, return_tensors="pt")

print("原始文本:", text)
# convert_ids_to_tokens：把数字ID转回文字，让我们能看懂分词结果
print("分词结果:", tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
# 文字对应的数字ID（模型真正的输入）
print("Token IDs:", tokens["input_ids"])
# 注意力掩码（告诉模型哪些是有效内容）
print("Attention Mask:", tokens["attention_mask"])
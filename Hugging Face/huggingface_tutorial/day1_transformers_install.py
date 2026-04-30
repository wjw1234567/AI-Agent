"""
Day1: Transformers 库安装与基础调用
"""

# 安装命令
# pip install transformers torch

import os

os.environ["HF_HUB_OFFLINE"] = "1"         # 强制离线（关键！）
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内加速！


from transformers import pipeline

# 基础调用：情感分析,多情感分类模型
#  model="jaylanyu/bert-base-chinese-emotion-classification" 这个是中文情感模型
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base"
                      ,local_files_only=True #只读本地模型
                      )
# result = classifier("I do not love learning Hugging Face!")
result = classifier("I have to learning Hugging Face! because i need to know ai agent")
print(result)
# 输出: [{'label': 'POSITIVE', 'score': 0.99...}]
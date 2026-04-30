"""
Day7: 复盘 - 知识点总结
"""

# ========== 核心概念回顾 ==========

# 1. Transformers 库
# - pipeline(): 快速调用预训练模型
# - AutoTokenizer / AutoModel: 自动加载对应模型

# 2. Tokenizer
# - 分词: text → tokens → token_ids
# - padding / truncation: 处理不同长度文本

# 3. 任务类型
# - 文本生成: text-generation
# - 对话: conversational  
# - 摘要: summarization
# - 情感分析: sentiment-analysis

# 4. Embedding
# - sentence-transformers: 生成句子向量
# - 余弦相似度: 计算文本相似性

# 5. Datasets
# - load_dataset(): 加载数据集
# - map(): 批量预处理
# - train_test_split(): 划分数据集

# 6. 实战应用
# - 文档向量化 + 相似度检索 = 简易问答系统基础

# ========== 进阶学习方向 ==========
# - Fine-tuning: 微调预训练模型
# - PEFT: 参数高效微调 (LoRA, QLoRA)
# - Transformers Trainer: 简化训练流程
# - 部署: ONNX, TensorFlow Lite
"""
Day6: 小 demo - 本地文档 → 向量化
"""

import os
from sentence_transformers import SentenceTransformer
import numpy as np

# 模拟本地文档（实际可读取 .txt/.md 文件）
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing deals with text data.",
    "Computer vision enables machines to understand images."
]

# 加载 embedding 模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 文档向量化
embeddings = model.encode(documents)
print(f"已向量化 {len(documents)} 个文档")
print(f"向量维度: {embeddings.shape}")

# 保存向量（实际应用可存入向量数据库）
np.save("document_embeddings.npy", embeddings)

# 查询示例
query = "What is Python?"
query_embedding = model.encode([query])

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity(query_embedding, embeddings)[0]

# 返回最相似的文档
top_idx = np.argmax(scores)
print(f"\n查询: {query}")
print(f"最相关文档: {documents[top_idx]}")
print(f"相似度: {scores[top_idx]:.4f}")
"""
Day4: 文本向量 Embedding 生成
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# 加载预训练 embedding 模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 生成句子向量
sentences = [
    "The cat is sleeping on the couch",
    "A dog is running in the park",
    "Cats and dogs are popular pets"
]

embeddings = model.encode(sentences)

print("句子数量:", len(sentences))
print("向量维度:", embeddings.shape)

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
print("句子1与句子3的余弦相似度:", similarity)
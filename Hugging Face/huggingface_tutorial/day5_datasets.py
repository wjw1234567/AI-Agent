"""
Day5: 数据集 Datasets 库加载与预处理
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# 加载内置数据集
dataset = load_dataset("imdb", split="train")
print("数据集大小:", len(dataset))
print("示例数据:", dataset[0])

# 划分训练/测试集
train_test = dataset.train_test_split(test_size=0.2)
print("训练集:", len(train_test["train"]))
print("测试集:", len(train_test["test"]))

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 预处理函数
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# 应用预处理
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("预处理后特征:", tokenized_dataset.features.keys())
# KNN近邻算法示例：预测鸢尾花种类

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载经典的鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]  # 只用前两个特征：花萼长度和宽度
y = iris.target
feature_names = ['花萼长度', '花萼宽度']
class_names = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 尝试不同的K值
print("=" * 50)
print("测试不同的K值")
print("=" * 50)
for k in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"K={k}: 准确率 {accuracy:.2%}")

# 4. 使用最佳K值（这里选K=5）
best_k = 5
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

print("\n" + "=" * 50)
print(f"最终模型 (K={best_k})")
print("=" * 50)
print(f"准确率: {accuracy_score(y_test, y_pred):.2%}")
print(f"\n详细报告:\n{classification_report(y_test, y_pred, target_names=class_names)}")

# 5. 预测新样本
new_flower = np.array([[5.0, 3.0], [6.5, 3.2], [7.0, 3.5]])
predictions = knn_model.predict(new_flower)
distances, indices = knn_model.kneighbors(new_flower)

print("\n新花朵预测：")
for i, (flower, pred) in enumerate(zip(new_flower, predictions)):
    print(f"花萼 [{flower[0]}, {flower[1]}] → 预测: {class_names[pred]}")
    print(f"  最近的{best_k}个邻居距离: {distances[i]}")

# 6. 可视化决策边界
plt.figure(figsize=(12, 5))

# 左图：训练数据
plt.subplot(1, 2, 1)
for i, class_name in enumerate(class_names):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
                label=class_name, alpha=0.6, s=50)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('训练数据分布')
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：测试结果
plt.subplot(1, 2, 2)

for i, class_name in enumerate(class_names):
    correct = (y_test == i) & (y_pred == i)
    incorrect = (y_test == i) & (y_pred != i)
    plt.scatter(X_test[correct, 0], X_test[correct, 1],
                label=f'{class_name}(正确)', marker='o', s=100, alpha=0.7)
    plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1],
                label=f'{class_name}(错误)', marker='x', s=100, linewidths=3)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title(f'KNN测试结果 (K={best_k})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Result/knn.png')
print("\n图表已保存到 knn.png")

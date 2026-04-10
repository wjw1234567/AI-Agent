# 决策树和随机森林示例：预测水果类型

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 准备数据：水果特征（重量g，直径cm）和类型（0=苹果，1=橙子，2=柠檬）
# 特征：[重量, 直径]
fruits = np.array([
    [150, 7], [160, 7.5], [170, 8], [155, 7.2], [165, 7.8],  # 苹果
    [200, 8], [210, 8.5], [220, 9], [205, 8.2], [215, 8.7],  # 橙子
    [80, 5], [85, 5.2], [90, 5.5], [82, 5.1], [88, 5.4]      # 柠檬
])
labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
fruit_names = ['苹果', '橙子', '柠檬']

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    fruits, labels, test_size=0.2, random_state=42
)

# 3. 决策树模型
print("=" * 50)
print("决策树模型")
print("=" * 50)
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"准确率: {dt_accuracy:.2%}")

# 4. 随机森林模型
print("\n" + "=" * 50)
print("随机森林模型")
print("=" * 50)
rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"准确率: {rf_accuracy:.2%}")

# 5. 特征重要性
print("\n特征重要性（随机森林）：")
for i, importance in enumerate(rf_model.feature_importances_):
    feature_name = ['重量', '直径'][i]
    print(f"{feature_name}: {importance:.2%}")

# 6. 预测新水果
new_fruits = np.array([[175, 7.5], [95, 5.3], [208, 8.4]])

print("\n决策树预测：")
for fruit, pred in zip(new_fruits, dt_model.predict(new_fruits)):
    print(f"重量{fruit[0]}g, 直径{fruit[1]}cm → {fruit_names[pred]}")

print("\n随机森林预测：")
for fruit, pred in zip(new_fruits, rf_model.predict(new_fruits)):
    print(f"重量{fruit[0]}g, 直径{fruit[1]}cm → {fruit_names[pred]}")

# 7. 可视化决策树
plt.figure(figsize=(15, 8))
plot_tree(dt_model, feature_names=['重量(g)', '直径(cm)'],
          class_names=fruit_names, filled=True, rounded=True, fontsize=10)
plt.title('决策树可视化')
plt.savefig('Result/tree_决策树可视化.png', dpi=100, bbox_inches='tight')
print("\n决策树图已保存到 tree.png")

# 8. 可视化分类结果
plt.figure(figsize=(10, 6))
for i, name in enumerate(fruit_names):
    mask = labels == i
    plt.scatter(fruits[mask, 0], fruits[mask, 1], label=name, s=100, alpha=0.6)

plt.scatter(new_fruits[:, 0], new_fruits[:, 1],
            c='black', marker='*', s=300, label='新预测', edgecolors='yellow', linewidths=2)
plt.xlabel('重量 (g)')
plt.ylabel('直径 (cm)')
plt.title('水果分类：决策树/随机森林')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Result/DecisionTree.png')
print("分类结果图已保存到 DecisionTree.png")

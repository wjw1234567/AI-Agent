# 逻辑回归示例：预测学生是否通过考试

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 准备数据：学习时间（小时）和是否通过考试（0=不通过，1=通过）
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 5, 7, 8]).reshape(-1, 1)
pass_exam = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1])

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    study_hours, pass_exam, test_size=0.2, random_state=42
)

# 3. 创建并训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 进行预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # 获取概率

# 5. 评估模型
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("逻辑回归模型结果")
print("=" * 50)
print(f"准确率: {accuracy:.2%}")
print(f"\n混淆矩阵:\n{conf_matrix}")
print(f"\n详细报告:\n{classification_report(y_test, y_pred, target_names=['不通过', '通过'])}")

# 6. 预测新数据
new_hours = np.array([[4], [7], [9]])
predictions = model.predict(new_hours)
probabilities = model.predict_proba(new_hours)

print("\n新学生预测：")
for hours, pred, prob in zip(new_hours, predictions, probabilities):
    result = "通过" if pred == 1 else "不通过"
    print(f"学习 {hours[0]} 小时 → 预测: {result} (通过概率: {prob[1]:.2%})")

# 7. 可视化
plt.figure(figsize=(10, 6))

# 绘制训练数据
colors_train = ['red' if label == 0 else 'green' for label in y_train]
plt.scatter(X_train, y_train, c=colors_train, alpha=0.6, s=100, label='训练数据')

# 绘制测试数据
colors_test = ['darkred' if label == 0 else 'darkgreen' for label in y_test]
plt.scatter(X_test, y_test, c=colors_test, marker='x', s=150, linewidths=3, label='测试数据')

# 绘制S型曲线
x_range = np.linspace(0, 11, 300).reshape(-1, 1)
y_prob_range = model.predict_proba(x_range)[:, 1]
plt.plot(x_range, y_prob_range, 'b-', linewidth=2, label='概率曲线')
plt.axhline(y=0.5, color='gray', linestyle='--', label='决策边界(50%)')

plt.xlabel('学习时间（小时）')
plt.ylabel('通过概率')
plt.title('逻辑回归：考试通过预测')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Result/LogisticRegression.png')
print("\n图表已保存到 LogisticRegression.png")

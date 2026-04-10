# 线性回归示例：预测房价

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 准备数据：房屋面积 vs 房价
# 假设数据：面积（平方米）和价格（万元）
house_sizes = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140]).reshape(-1, 1)
house_prices = np.array([150, 180, 210, 240, 270, 300, 330, 360, 390, 420])

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    house_sizes, house_prices, test_size=0.2, random_state=42
)

# 3. 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 进行预测
y_pred = model.predict(X_test)

# 5. 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("线性回归模型结果")
print("=" * 50)
print(f"模型公式: 价格 = {model.coef_[0]:.2f} × 面积 + {model.intercept_:.2f}")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"R² 分数: {r2:.4f}")
print("\n预测示例：")
print(f"测试集实际价格: {y_test}")
print(f"测试集预测价格: {y_pred}")

# 6. 预测新数据
new_size = np.array([[95]])
predicted_price = model.predict(new_size)
print(f"\n新房子面积 {new_size[0][0]} 平方米，预测价格: {predicted_price[0]:.2f} 万元")

# 7. 可视化
plt.figure(figsize=(10, 6))
plt.scatter(house_sizes, house_prices, color='blue', label='实际数据')
plt.plot(house_sizes, model.predict(house_sizes), color='red', linewidth=2, label='拟合直线')
plt.scatter(X_test, y_pred, color='green', marker='x', s=100, label='测试预测')
plt.xlabel('房屋面积（平方米）')
plt.ylabel('房价（万元）')
plt.title('线性回归：房价预测')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Result/LinearRegression.png')
print("\n图表已保存到 result.png")

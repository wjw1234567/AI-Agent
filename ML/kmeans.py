# K-Means聚类示例：客户分群

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 准备数据：客户的年收入（万元）和年消费（万元）
customers = np.array([
    [30, 10], [35, 12], [40, 15], [32, 11],  # 低收入低消费
    [80, 40], [85, 45], [90, 50], [88, 48],  # 中等收入中等消费
    [150, 80], [160, 85], [155, 82], [165, 88],  # 高收入高消费
    [45, 8], [50, 10], [55, 12],  # 低收入低消费
    [95, 55], [100, 60], [92, 52],  # 中等收入中等消费
    [170, 90], [175, 95]  # 高收入高消费
])

# 2. 数据标准化（让不同特征在同一尺度）
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

# 3. 使用肘部法则找最佳K值
inertias = []
K_range = range(1, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(customers_scaled)
    inertias.append(kmeans.inertia_)

# 4. 使用K=3进行聚类
k = 3
kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans_model.fit_predict(customers_scaled)

# 5. 获取聚类中心（转换回原始尺度）
centers_scaled = kmeans_model.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

print("=" * 50)
print(f"K-Means聚类结果 (K={k})")
print("=" * 50)

# 6. 分析每个聚类
cluster_names = ['低消费群体', '中等消费群体', '高消费群体']
for i in range(k):
    cluster_customers = customers[clusters == i]
    print(f"\n聚类 {i} ({cluster_names[i]}):")
    print(f"  客户数量: {len(cluster_customers)}")
    print(f"  平均收入: {cluster_customers[:, 0].mean():.1f} 万元")
    print(f"  平均消费: {cluster_customers[:, 1].mean():.1f} 万元")
    print(f"  聚类中心: 收入{centers[i, 0]:.1f}万, 消费{centers[i, 1]:.1f}万")

# 7. 预测新客户属于哪个群体
new_customers = np.array([[60, 25], [120, 65], [40, 12]])
new_customers_scaled = scaler.transform(new_customers)
new_clusters = kmeans_model.predict(new_customers_scaled)

print("\n新客户分群：")
for customer, cluster in zip(new_customers, new_clusters):
    print(f"收入{customer[0]}万, 消费{customer[1]}万 → {cluster_names[cluster]}")

# 8. 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：肘部法则
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('聚类数量 K')
axes[0].set_ylabel('簇内误差平方和')
axes[0].set_title('肘部法则：选择最佳K值')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=3, color='red', linestyle='--', label='选择的K=3')
axes[0].legend()

# 右图：聚类结果
colors = ['red', 'blue', 'green']

for i in range(k):
    cluster_points = customers[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[i], label=cluster_names[i], s=100, alpha=0.6)

# 绘制聚类中心
plt.scatter(centers[:, 0], centers[:, 1],
            c='black', marker='*', s=500, edgecolors='yellow',
            linewidths=2, label='聚类中心')

# 绘制新客户
plt.scatter(new_customers[:, 0], new_customers[:, 1],
            c='purple', marker='D', s=150, edgecolors='black',
            linewidths=2, label='新客户')

axes[1].set_xlabel('年收入（万元）')
axes[1].set_ylabel('年消费（万元）')
axes[1].set_title('客户聚类结果')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Result/kmeans.png')
print("\n图表已保存到 result.png")

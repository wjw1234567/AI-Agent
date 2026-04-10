# 朴素贝叶斯示例：垃圾邮件分类

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 准备数据：邮件内容和标签（0=正常，1=垃圾）
emails = [
    "恭喜中奖一百万 立即领取",
    "免费获得iPhone 点击链接",
    "明天开会讨论项目进度",
    "周末一起吃饭吗",
    "你的账户异常 请立即验证",
    "会议纪要已发送 请查收",
    "限时优惠 买一送一",
    "项目文档已更新",
    "免费赠送礼品 快来领取",
    "下周一提交报告",
    "恭喜获得大奖 点击领取",
    "今天下班一起打球",
    "您的快递已到 请签收",
    "紧急通知 账户被冻结",
    "周报已完成 请审阅",
    "免费试用 立即注册",
    "年度总结报告",
    "中奖通知 请查收",
    "团队建设活动通知",
    "优惠券限时领取"
]
labels = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])

# 2. 文本向量化（把文字转成数字）
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# 4. 创建并训练朴素贝叶斯模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. 进行预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 6. 评估模型
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("朴素贝叶斯模型结果")
print("=" * 50)
print(f"准确率: {accuracy:.2%}")
print(f"\n混淆矩阵:\n{conf_matrix}")
print(f"\n详细报告:\n{classification_report(y_test, y_pred, target_names=['正常邮件', '垃圾邮件'], zero_division=0)}")

# 7. 预测新邮件
new_emails = [
    "恭喜您中奖了 请点击链接",
    "明天的会议改到下午三点",
    "免费送手机 限时优惠"
]
new_X = vectorizer.transform(new_emails)
new_pred = model.predict(new_X)
new_prob = model.predict_proba(new_X)

print("\n新邮件预测：")
for email, pred, prob in zip(new_emails, new_pred, new_prob):
    result = "垃圾邮件" if pred == 1 else "正常邮件"
    confidence = prob[pred]
    print(f"'{email}'")
    print(f"  → {result} (置信度: {confidence:.2%})\n")

# 8. 查看最重要的词
feature_names = vectorizer.get_feature_names_out()

# 找出垃圾邮件中最常见的词
spam_prob = np.exp(model.feature_log_prob_[1])  # 类别1（垃圾邮件）
top_spam_indices = spam_prob.argsort()[-5:][::-1]

print("\n垃圾邮件中最常见的词：")
for idx in top_spam_indices:
    print(f"  '{feature_names[idx]}': {spam_prob[idx]:.2%}")

print("\n" + "=" * 50)
print("运行完成！")

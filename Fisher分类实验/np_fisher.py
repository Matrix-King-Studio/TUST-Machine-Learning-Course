import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 加载MNIST数据集
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target

# 将标签转换为二分类：'1' 或 '非1'
y_bin = np.where(y == '1', 1, 0)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)


def fisher_lda(X, y):
    # 1. 计算各类样本均值向量
    m1 = np.mean(X[y == 1], axis=0)
    m2 = np.mean(X[y == 0], axis=0)

    # 2. 计算样本类内离散度矩阵
    S1 = np.dot((X[y == 1] - m1).T, (X[y == 1] - m1))
    S2 = np.dot((X[y == 0] - m2).T, (X[y == 0] - m2))
    Sw = S1 + S2

    # 4. 求投影方向向量W  奇异矩阵求伪逆
    w = np.linalg.pinv(Sw).dot(m1 - m2)

    return w


# 训练Fisher的LDA
w = fisher_lda(X_train, y_train)

# 5. 将训练集内所有样本进行投影
y_train_proj = X_train.dot(w)
y_test_proj = X_test.dot(w)

# 6. 计算在投影空间上的分割阈值y0
m1_tilde = np.mean(y_train_proj[y_train == 1])
m2_tilde = np.mean(y_train_proj[y_train == 0])
N1 = np.sum(y_train == 1)
N2 = np.sum(y_train == 0)
y0 = (N1 * m1_tilde + N2 * m2_tilde) / (N1 + N2)

# 7. 对于给定的测试样本x，计算出它在w上的投影点y
# 已完成在上面的 y_test_proj

# 8. 根据决策规则进行分类
y_test_pred = np.where(y_test_proj > y0, 1, 0)

# 计算分类准确率
accuracy = np.mean(y_test_pred == y_test)
print(f"分类准确率: {accuracy:.4f}")

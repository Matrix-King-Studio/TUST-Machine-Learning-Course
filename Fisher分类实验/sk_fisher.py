# Fisher线性判别分析-Fisher LDA
# 手写数字识别的Fisher十分类Python程序
import numpy as np
from sklearn import discriminant_analysis
from sklearn import datasets
from sklearn.metrics import accuracy_score

np.random.seed(1000)

digit = datasets.load_digits()
# 10类，每类180左右，共1797个样本，64维
digit_x = digit.data
digit_y = digit.target
indices = np.random.permutation(len(digit_x))
# 留200个样本作为测试集，剩下是训练集（大概是9：1）
digit_x_train = digit_x[indices[:-200]]
digit_y_train = digit_y[indices[:-200]]
digit_x_test = digit_x[indices[-200:]]
digit_y_test = digit_y[indices[-200:]]
# 定义Fisher分类器对象fisher_clf
fisher_clf = discriminant_analysis.LinearDiscriminantAnalysis()
# 调用该对象的训练方法
fisher_clf.fit(digit_x_train, digit_y_train)
# 调用该对象的测试方法
digit_y_pred = fisher_clf.predict(digit_x_test)
print('测试数据集的正确标签为:', digit_y_test)
print('测试数据集的预测标签为:', digit_y_pred)

testing_acc = accuracy_score(digit_y_test, digit_y_pred) * 100
print('Fisher线性分类器测试准确率: {:.2f}%'.format(testing_acc))

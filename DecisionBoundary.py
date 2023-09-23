import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 加载iris数据集
iris = load_iris()
X = iris.data[:, 2:]  # 仅使用后两个特征以便于可视化
y = iris.target

# 训练决策树模型
clf = DecisionTreeClassifier().fit(X, y)

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Decision Boundary of Decision Tree')

# 显示图形
plt.show()

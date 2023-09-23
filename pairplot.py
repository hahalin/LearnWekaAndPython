import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# 加载iris数据集
iris = load_iris()
data = iris.data
feature_names = iris.feature_names
target = iris.target

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=feature_names)
df['species'] = target

# 创建散点图矩阵
sns.pairplot(df, hue='species', markers=["o", "s", "D"])

# 显示图形
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 加载iris数据集
iris = load_iris()
data = iris.data
feature_names = iris.feature_names
target = iris.target

# 将数据转换为DataFrame
import pandas as pd
df = pd.DataFrame(data, columns=feature_names)
df['species'] = target

# 计算相关性矩阵
corr_matrix = df.corr()

# 'viridis'：一种渐变的紫色到黄色的颜色映射。
# 'plasma'：一种渐变的紫色到橙色的颜色映射。
# 'inferno'：一种渐变的黑色到黄色的颜色映射。
# 'magma'：一种渐变的黑色到粉红色的颜色映射。
# 'cividis'：一种渐变的深蓝色到黄色的颜色映射。

# 创建热力图
plt.figure(figsize=(10, 8))
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
sns.heatmap(corr_matrix, annot=True, cmap='inferno', fmt=".2f")

# 显示图形
plt.show()

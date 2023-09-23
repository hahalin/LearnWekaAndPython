import os
import cv2
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

dot_data = export_graphviz(clf, out_file=None, 
                            feature_names=iris.feature_names,  
                            class_names=iris.target_names,  
                            filled=True, rounded=True,  
                            special_characters=True)

graph = graphviz.Source(dot_data)
graph.render(filename='iris_tree', format='png', cleanup=True)  # 保存为PNG

img = mpimg.imread('iris_tree.png')

# plt.figure(figsize=(20, 20))
# plt.imshow(img)
#plt.axis('off')
#plt.show()

cv2.imshow('Image', img)

# 等待用户按键，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


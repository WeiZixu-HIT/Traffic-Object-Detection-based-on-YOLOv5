import os
import matplotlib.pyplot as plt
import numpy as np

# x = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'trafficlight', 'stopsign', 'all classes']
#map50
# init = [86.5, 71.1, 91.8, 81.3, 80.2, 79.2, 84.3, 81.2, 82.0]    #map50
# change = [86.1, 72.5, 91.8, 82.5, 80.3, 79.1, 84.4, 81.4, 82.3]

#map50-95
m = [59.1, 43.8, 65.5, 50.4, 64.1, 61.5, 51.8, 65.5, 57.7]

m1 = [59.7, 45.8, 66.2, 51.8, 64.7, 62.3, 52.7, 66.6, 58.7]

# print(np.mean(init))
# print(np.mean(change))

#precision recall
x = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'trafficlight', 'stopsign', 'all classes']
p_init = [83.5, 73.2, 83.0, 81.0, 82.7, 75.6, 80.1, 77.2, 79.5]
p_change = [87.2, 89.3, 62.1, 89.8, 86.3, 69.1, 80.2, 98.7, 82.8]
r_init = [76.7, 61.8, 85.0, 72.5, 66.7, 70.0, 74.1, 68.9, 72.0]
r_change = [82.0, 67.0, 87.0, 79.0, 63.0, 67.0, 81.0, 77.0, 75.3]

#2、创建画布
plt.figure(figsize = (10,10),dpi=80)
#3、绘制柱状图
x_ticks = range(len(x))
plt.bar(x, m, width=0.2,label='改进前')
plt.bar([i+0.2 for i in x_ticks],m1,width=0.2,label='改进后')
#4、修改X刻度
plt.xticks(x_ticks,x, rotation=15)
#添加网格显示
plt.grid(linestyle='--',alpha=0.5)
#5、标题
plt.title("各类别及总的mAP@50-95对比图", size = 20)
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.tick_params(labelsize=15)
plt.legend(loc='best')
plt.xlabel('类别', size=18)
plt.ylabel('IoU阈值从50%-95%的均值平均精确度 / %', size=18)
plt.savefig('runs/newpic/fenlei_mAP50-95图.png')
plt.show()
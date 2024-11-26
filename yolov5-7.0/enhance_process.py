import os
import matplotlib.pyplot as plt
import numpy as np
# # path = 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/labels/train'
# # txts = os.listdir(path)
# # person, bicycle, car, motorcycle, bus, truck, trafficlight, stopsign = 0, 0, 0, 0, 0, 0, 0, 0
# # for txt in txts:
# #     f = open(path + '/' + txt, 'r')
# #     contents = f.readlines()
# #     f.close()
# #     for content in contents:
# #         cls = content.strip('\n').split(' ')[0]
# #         if cls == '0':
# #             person += 1
# #         elif cls == '1':
# #             bicycle += 1
# #         elif cls == '2':
# #             car += 1
# #         elif cls == '3':
# #             motorcycle += 1
# #         elif cls == '4':
# #             bus += 1
# #         elif cls == '5':
# #             truck += 1
# #         elif cls == '6':
# #             trafficlight += 1
# #         elif cls == '7':
# #             stopsign += 1
# # print('person 有%d 例'%person)
# # print('bicycle 有%d 例'%bicycle)
# # print('car 有%d 例'%car)
# # print('motorcycle 有%d 例'%motorcycle)
# # print('bus 有%d 例'%bus)
# # print('truck 有%d 例'%truck)
# # print('trafficlight 有%d 例'%trafficlight)
# # print('stopsign 有%d 例'%stopsign)
# # #
# # x = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'trafficlight', 'stopsign']
# # y = [person, bicycle, car, motorcycle, bus, truck, trafficlight, stopsign]
# # #绘制条形图
# # plt.figure(figsize = (10,10),dpi=80)
# # plt.bar(x,y)
# # #设置刻度及刻度标签
# # x_t = list(range(len(x)))
# # plt.xticks(x_t,x,rotation=16)
# # plt.tick_params(labelsize=20)
# # plt.xlabel('类别',size = 20)
# # plt.ylabel('数量/个', size = 20)
# # plt.title("数据集实例分布图", size = 20)
# # # # 支持中文
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# # plt.savefig('dataset/数据集实例分布new.png')
# # plt.show()
#
# # path1 = 'C:/PythonProject/yolov5-7.0/yolov5-7.0/runs/detect/labels'
# # path2 = 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/enhance_jpg'
# # txtnames = os.listdir(path1)
# # imgnames = os.listdir(path2)
# # for i in imgnames:
# #     prename = i.split('.jpg')[-2]
# #     labelname = prename + '.txt'
# #     if labelname not in txtnames:
# #         os.remove('C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/enhance_jpg/' + prename + '.jpg')
# #         print(prename + '.jpg' + '已被删除')
#
# # x = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'trafficlight', 'stopsign', 'all classes']
# # init = [91.2, 80.9, 93.9, 87.9, 84.7, 83.7, 88.6, 89.6, 87.6]    #map50
# # change = [93.1, 82.5, 95.9, 89.1, 86.0, 84.9, 90.1, 90.9, 89.1]
# # init = [91.2, 80.9, 93.9, 87.9, 84.7, 83.7, 88.6, 89.6]
# # change = [93.1, 82.5, 95.9, 89.1, 86.0, 84.9, 90.1, 90.9]
# # print(np.mean(init))
# # print(np.mean(change))
#
# #precision recall
# x = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'trafficlight', 'stopsign', 'all classes']
# # p_init = [86.1, 77.7, 85.5, 84.3, 84.2, 79.3, 83.0, 87.9, 83.5]
# # p_change = [87.5, 78.9, 84.1, 86.1, 86.8, 81.7, 83.9, 89.4, 84.8]
# r_init = [83.7, 71.4, 87.7, 79.6, 70.9, 74.6, 78.8, 78.0, 78.1]
# r_change = [85.0, 72.5, 88.9, 80.8, 72.1, 75.7, 80.1, 79.2, 79.3]
# # for i in range(9):
# #     print(x[i] + '类提升了 %.1f '%(r_change[i] - r_init[i]))
# # x = ['lane0', 'lane1', 'lane2', 'lane3', 'lane4', 'lane5', 'lane6', 'all']
# # y = [72.9, 68.5, 82.3, 77.5, 85.5, 76.0, 90.2, 79.0]
# # z = [99.6, 99.3, 98.8, 99.5, 98.0, 99.3, 99.7, 99.2]
# # y = [72.5, 67.9, 82.6, 77.6, 86.0, 75.3, 86.3, 78.3]
# # z = [99.1, 98.5, 99.3, 99.6, 98.8, 98.5, 95.6, 98.5]
# # y = [64.8, 59.6, 82.5, 76.4, 82.9, 71.5, 88.1, 75.1]
# # z = [99.5, 99.4, 99.4, 99.3, 99.4, 99.2, 99.2, 99.4]
# # #2、创建画布
# # plt.figure(figsize = (10,10),dpi=80)
# # #3、绘制柱状图
# # x_ticks = range(len(x))
# # plt.bar(x, y, width=0.2,label='Mask mAP@50')
# # plt.bar([i+0.2 for i in x_ticks],z,width=0.2,label='Box mAP@50')
# # #4、修改X刻度
# # plt.xticks(x_ticks,x, rotation=0)
# # #添加网格显示
# # plt.grid(linestyle='--',alpha=0.5)
# # #5、标题
# # plt.title("各类别mAP@50", size = 20)
# # # 支持中文
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# # plt.tick_params(labelsize=15)
# # plt.legend(loc='best')
# # plt.xlabel('类别', size=18)
# # plt.ylabel('mAP@50 / %', size=18)
# # plt.savefig('runs/paperpic/分割mAP@50图.png')
# # plt.show()
#
# #2、创建画布
# plt.figure(figsize = (10,10),dpi=80)
# #3、绘制柱状图
# x_ticks = range(len(x))
# plt.bar(x, r_init, width=0.2,label='改进前')
# plt.bar([i+0.2 for i in x_ticks],r_change,width=0.2,label='改进后')
# #4、修改X刻度
# plt.xticks(x_ticks,x, rotation=15)
# #添加网格显示
# plt.grid(linestyle='--',alpha=0.5)
# #5、标题
# plt.title("各类别及总的召回率对比图", size = 20)
# # 支持中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.tick_params(labelsize=15)
# plt.legend(loc='best')
# plt.xlabel('类别', size=18)
# plt.ylabel('召回率 / %', size=18)
# plt.savefig('runs/paperpic/new_fenlei_recall图.png')
# plt.show()

x = ['预处理', '推理', '非极大值抑制', '总耗时']
y = [0.2, 10.6, 1.0, 11.8]
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.bar(x, y)
plt.xlabel('阶段')
plt.ylabel('耗时 / ms')
plt.title('各阶段耗时')
plt.savefig('runs/newpic/FPS图.png')
plt.show()

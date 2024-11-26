import pandas as pd
import matplotlib.pyplot as plt

path = ['C:/PythonProject/yolov5-7.0/yolov5-7.0/runs/train/duizhaozu/runs/train/duizhaozu/results.csv', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/runs/train/shiyanzu/runs/train/shiyanzu/results.csv']
labels = ['改进前', '改进后']
x = []
for i in range(100):
    x.append(i)
plt.figure(figsize = (15,10),dpi=80)

for j in range(len(path)):
    data = pd.read_csv(path[j])    #data为结构体
    m = data['     metrics/mAP_0.5'].tolist()[0:100]
    # if j == 1:
    #
    #     for i, k in enumerate(m):
    #         m[i] = k + 0.02
    m1 = data['metrics/mAP_0.5:0.95'].tolist()[0:100]
    # if j == 1:
    #     for i, k in enumerate(m1):
    #         m1[i] = k + 0.01
    p = data['   metrics/precision'].tolist()[0:100]
    # if j == 1:
    #     for i, k in enumerate(p):
    #         p[i] = k + 0.02
    r = data['      metrics/recall'].tolist()[0:100]
    vb = data['        val/box_loss'].tolist()[0:100]
    vo = data['        val/obj_loss'].tolist()[0:100]
    vc = data['        val/cls_loss'].tolist()[0:100]
    # if j == 1:
    #     for i, k in enumerate(vc):
    #         vc[i] = k - 0.0006

    plt.subplot(241)
    plt.plot(x,m, label = labels[j])
    plt.title('mAP@50对比图')
    # plt.rcParams['xtick.labelsize'] = 24
    # plt.rcParams['ytick.labelsize'] = 24
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.xlim(0, 100)
    plt.ylabel('mAP@50')
    plt.ylim(0.80, 0.825)

    plt.subplot(242)
    plt.plot(x, m1, label=labels[j])
    plt.title('mAP@50-95对比图')
    # plt.rcParams['xtick.labelsize'] = 24
    # plt.rcParams['ytick.labelsize'] = 24
    # # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.xlim(0, 100)
    plt.ylabel('mAP@50-95')
    plt.ylim(0.5, 0.59)

    plt.subplot(243)
    plt.plot(x, p, label=labels[j])
    plt.title('精确率对比图')
    # plt.rcParams['xtick.labelsize'] = 24
    # plt.rcParams['ytick.labelsize'] = 24
    # # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.xlim(0, 100)
    plt.ylabel('精确率')
    plt.ylim(0.74, 0.82)

    plt.subplot(244)
    plt.plot(x, r, label=labels[j])
    plt.title('召回率对比图')
    # plt.rcParams['xtick.labelsize'] = 24
    # plt.rcParams['ytick.labelsize'] = 24
    # # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.xlim(0, 100)
    plt.ylabel('召回率')
    plt.ylim(0.65, 0.73)

    plt.subplot(245)
    plt.plot(x, vb, label=labels[j])
    plt.title('验证集定位误差对比图')
    # plt.rcParams['xtick.labelsize'] = 24
    # plt.rcParams['ytick.labelsize'] = 24
    # # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.xlim(0, 100)
    plt.ylabel('验证集定位误差')
    plt.ylim(0.025, 0.08)
    #
    plt.subplot(246)
    plt.plot(x, vo, label=labels[j])
    plt.title('验证集置信度误差对比图')
    # plt.rcParams['xtick.labelsize'] = 24
    # plt.rcParams['ytick.labelsize'] = 24
    # # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.xlim(0, 100)
    plt.ylabel('验证集置信度误差')
    plt.ylim(0.023, 0.0275)
    #
    plt.subplot(247)
    plt.plot(x, vc, label=labels[j])
    plt.title('验证集分类误差对比图')
    # plt.rcParams['xtick.labelsize'] = 24
    # plt.rcParams['ytick.labelsize'] = 24
    # # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.xlim(0, 100)
    plt.ylabel('验证集分类误差')
    plt.ylim(0.007, 0.010)

plt.tight_layout()
plt.savefig('C:/PythonProject/yolov5-7.0/yolov5-7.0/runs/newpic/对比图.png')
plt.show()
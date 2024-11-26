import os
# paths = ['C:/PythonProject/yolov5-7.0/yolov5-7.0/rightlabels/test', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/rightlabels/train', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/rightlabels/val']
# for path in paths:
#     txtnames = os.listdir(path)
#     num0, num7, num9 = 0, 0, 0
#     for txtname in txtnames:
#         f = open(path + '/' + txtname, 'r')
#         txt = f.readlines()
#         # print('原始内容:')
#         # print(txt)
#         f.close()
#         for k in txt:
#                 oldcontent = k.strip('\n').split()
#                 cls = int(oldcontent[0])
#                 if cls == 0:
#                     num0 = num0 + 1
#                     print('包含类别 0 的标注文件名为：' + txtname)
#                 if cls == 7:
#                     num7 = num7 + 1
#                     print('包含类别 7 的标注文件名为：' + txtname)
#                 if cls == 9:
#                     num9 = num9 + 1
#                     print('包含类别 9 的标注文件名为：' + txtname)
#     print(path.split('/')[-1] + '文件夹下包含类别 0 的标注文件共有' + str(num0) + '个')
#     print(path.split('/')[-1] + '文件夹下包含类别 7 的标注文件共有' + str(num7) + '个')
#     print(path.split('/')[-1] + '文件夹下包含类别 9 的标注文件共有' + str(num9) + '个')

paths = ['C:/PythonProject/yolov5-7.0/yolov5-7.0/rightlabels/train', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/rightlabels/test', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/rightlabels/val']
for path in paths:
    txtnames = os.listdir(path)
    for txtname in txtnames:
        f = open(path + '/' + txtname, 'r')
        txt = f.readlines()
        # print('原始内容:')
        # print(txt)
        f.close()
        newcontent = []
        for k in txt:
            oldcontent = k.strip('\n').split(' ', 1)
            cls, loc = int(oldcontent[0]), oldcontent[1]
            if cls == 8:
                newcls = 6
            else:
                newcls = cls - 1
            newcontent.append(str(newcls) + ' ' + loc + '\n')
        f = open(path + '/' + txtname, 'w')
        for line in newcontent:
            f.writelines(line)
        f.close()
        print(path + '/' + txtname + '已完成修改')

# newcontent = []
# f = open('C:/PythonProject/yolov5-7.0/yolov5-7.0/0_Road001_Trim003_frames_00024.txt', 'r')
# txt = f.readlines()
# f.close()
# for k in txt:
#     oldcontent = k.strip('\n').split(' ', 1)
#     cls, loc = int(oldcontent[0]), oldcontent[1]
#     print(cls)
#     print(loc)
#     if cls == 8:
#         newcls = 6
#     else:
#         newcls = cls - 1
#     newcontent.append(str(newcls) + ' ' + loc + '\n')
# print(newcontent)
# f = open('C:/PythonProject/yolov5-7.0/yolov5-7.0/0_Road001_Trim003_frames_00024.txt', 'w')
# for line in newcontent:
#     f.writelines(line)
# f.close()


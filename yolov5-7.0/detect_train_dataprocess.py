import os
# path = 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/Talk2Car/data/imgs/labels'
# path2 = 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/Talk2Car/data/imgs/imgs'
# txtnames = os.listdir(path)
# imgnames = os.listdir(path2)
# for i in imgnames:
#     prename = i.split('.jpg')[-2]
#     labelname = prename + '.txt'
#     if labelname not in txtnames:
#         print(labelname)
#         os.remove('C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/Talk2Car/data/imgs/imgs/' + prename + '.jpg')
#         print(prename + '.jpg' + '已被删除')
# paths = ['C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/labels/test', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/labels/train', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/labels/val']
# for path in paths:
#     txtnames = os.listdir(path)
#     for txtname in txtnames:
#         f = open(txtname)

# path = 'C:/PythonProject/yolov5-7.0/yolov5-7.0/runs/detect/labels'
# txtnames = os.listdir(path)
# for txtname in txtnames:
#     f = open(path + '/' + txtname, 'r')
#     txt = f.readlines()
#     print(txt)
#     f.close()
#     newcontent = []
#     for k in txt:
#         oldcontent = k.strip('\n').split(' ', 1)
#         cls, loc = oldcontent[0], oldcontent[1]
#         # cls = int(oldcontent[0])
#         # loc = oldcontent[1] + ' ' + oldcontent[2] + ' ' + oldcontent[3] + ' ' + oldcontent[4]
#         if cls == '1':
#             newcls = '1'
#         elif cls == '3':
#             newcls = '3'
#         elif cls == '5':
#             newcls = '4'
#         elif cls == '7':
#             newcls = '5'
#         elif cls == '9':
#             newcls = '6'
#         elif cls == '11':
#             newcls = '7'
#         else:
#             continue
#         newcontent.append(newcls + ' ' + loc + '\n')
#     print(newcontent)
#     f = open(path + '/' + txtname, 'w')
#     for line in newcontent:
#         f.writelines(line)
#
#     f.close()


# paths = ['C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/labels/test', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/labels/train', 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/labels/val']
# for path in paths:
#     txtnames = os.listdir(path)
#     for txtname in txtnames:
#         f = open(path + '/' + txtname, 'r')
#         txt = f.readlines()
#         print('原始内容:')
#         print(txt)
#         f.close()
#         newcontent = []
#         for k in txt:
#             oldcontent = k.strip('\n').split()
#             cls = int(oldcontent[0])
#             loc = oldcontent[1] + ' ' + oldcontent[2] + ' ' + oldcontent[3] + ' ' + oldcontent[4]
#             if cls == 5:
#                 newcls = '4'
#             elif cls == 7:
#                 newcls = '5'
#             elif cls == 9:
#                 newcls = '6'
#             elif cls == 11:
#                 newcls = '7'
#             else:
#                 newcls = str(cls)
#             newcontent.append(newcls + ' ' + loc + '\n')
#         print('修改后内容:')
#         print(newcontent)
#         f = open(path + '/' + txtname, 'w')
#         for line in newcontent:
#             f.writelines(line)
#
#         f.close()
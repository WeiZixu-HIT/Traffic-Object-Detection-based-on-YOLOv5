import os
import shutil

def split_file(src, dst1, dst2):
    '''
    function: 将文件中不同后缀的文件分开到不同文件夹
    example: 区分jpg和txt
    src:str(filefolder)
    dst:str(filefolder)
    '''
    #区分jpg和txt
    jpg = []
    txt = []
    for f in os.listdir(src):
        if f.endswith('.jpg'):
            jpg.append(f)
        elif f.endswith('.txt'):
            txt.append(f)
    #创建目标文件夹
    if not os.path.isdir(dst1):
        os.mkdir(dst1)
    if not os.path.isdir(dst2):
        os.mkdir(dst2)
    #拷贝文件到目标文件夹
    for j in jpg:
        _jpg = os.path.join(src, j)
        shutil.copy(_jpg, dst1)
    for t in txt:
        _txt = os.path.join(src, t)
        shutil.copy(_txt, dst2)
 #example
if __name__ == '__main__':
    base_filename = 'C:/PythonProject/yolov5-7.0/yolov5-7.0/dataset/'
    src = os.path.join(base_filename, 'enhance')
    dst1 = os.path.join(base_filename, 'enhance_jpg')
    dst2 = os.path.join(base_filename, 'enhance_txt')
    split_file(src, dst1, dst2)
    print('已将 enhance 文件夹下的 jpg文件 和 txt文件 分离！')


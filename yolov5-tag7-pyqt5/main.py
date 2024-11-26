import os
import sys
from pathlib import Path
import cv2
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap

from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
from utils.augmentations import letterbox
from utils.plots import plot_one_box

from models.common import DetectMultiBackend


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = False

        name = 'exp'
        save_file = ROOT / 'result'
        self.save_file = increment_path(Path(save_file) / name, exist_ok=False, mkdir=True)

        cudnn.benchmark = True
        weights = 'weights/voc_hat.pt'   # 模型加载路径
        imgsz = 640  # 预测图尺寸大小
        self.conf_thres = 0.25  # NMS置信度
        self.iou_thres = 0.45  # IOU阈值

        # 载入模型
        self.model = DetectMultiBackend(weights, device=self.device)
        stride = self.model.stride
        self.imgsz = check_img_size(imgsz, s=stride)
        if self.half:
            self.model.half()  # to FP16

        # 从模型中获取各类别名称
        self.names = self.model.names
        # 给每一个类别初始化颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("#centralwidget{border-image:url(./UI/hutao.jpg)}")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        # self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")


        # 打开单图片按钮
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(150, 40))
        self.pushButton_img.setMaximumSize(QtCore.QSize(150, 40))
        self.pushButton_img.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(12)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(self.pushButton_img, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(5)  # 增加垂直盒子内部对象间距
        self.pushButton_img.setToolTip('<b>请选择一张图片进行检测</b>')  # 创建提示框

        # 打开多图片按钮
        self.pushButton_imgs = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_imgs.sizePolicy().hasHeightForWidth())
        self.pushButton_imgs.setSizePolicy(sizePolicy)
        self.pushButton_imgs.setMinimumSize(QtCore.QSize(150, 40))
        self.pushButton_imgs.setMaximumSize(QtCore.QSize(150, 40))
        self.pushButton_imgs.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(12)
        self.pushButton_imgs.setFont(font)
        self.pushButton_imgs.setObjectName("pushButton_imgs")
        self.verticalLayout.addWidget(self.pushButton_imgs, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(5)  # 增加垂直盒子内部对象间距
        self.pushButton_imgs.setToolTip('<b>请选择一张或多张图片进行检测</b>')  # 创建提示框

        # 打开图片文件按钮
        self.pushButton_imgfile = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_imgfile.sizePolicy().hasHeightForWidth())
        self.pushButton_imgfile.setSizePolicy(sizePolicy)
        self.pushButton_imgfile.setMinimumSize(QtCore.QSize(150, 40))
        self.pushButton_imgfile.setMaximumSize(QtCore.QSize(150, 40))
        self.pushButton_imgfile.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(12)
        self.pushButton_imgfile.setFont(font)
        self.pushButton_imgfile.setObjectName("pushButton_imgfile")
        self.verticalLayout.addWidget(self.pushButton_imgfile, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(5)  # 增加垂直盒子内部对象间距
        self.pushButton_imgfile.setToolTip('<b>请选择包含所有检测图片的文件夹</b>')  # 创建提示框

        # 打开摄像头按钮
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(150, 40))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(150, 40))
        self.pushButton_camera.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(5)
        self.pushButton_camera.setToolTip('<b>请确保摄像头设备正常</b>')  # 创建提示框

        # 打开视频按钮
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(150, 40))
        self.pushButton_video.setMaximumSize(QtCore.QSize(150, 40))
        self.pushButton_video.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.pushButton_video, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(50)
        self.pushButton_video.setToolTip('<b>请选择一个视频进行检测</b>')  # 创建提示框

        # 显示导出文件夹按钮
        self.pushButton_showdir = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_showdir.sizePolicy().hasHeightForWidth())
        self.pushButton_showdir.setSizePolicy(sizePolicy)
        self.pushButton_showdir.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_showdir.setMaximumSize(QtCore.QSize(150, 50))
        self.pushButton_showdir.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        self.pushButton_showdir.setFont(font)
        self.pushButton_showdir.setObjectName("pushButton_showdir")
        self.verticalLayout.addWidget(self.pushButton_showdir, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_showdir.setToolTip('<b>检测后的文件将在此保存</b>')  # 创建提示框

        # 右侧图片/视频填充区域
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.label.setStyleSheet("border: 1px solid white;")  #  添加显示区域边框

        # 底部美化导航条
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YOLOv5目标检测平台"))
        self.pushButton_img.setText(_translate("MainWindow", "单图片检测"))
        self.pushButton_imgs.setText(_translate("MainWindow", "多图片检测"))
        self.pushButton_imgfile.setText(_translate("MainWindow", "文件夹图片检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.pushButton_showdir.setText(_translate("MainWindow", "显示输出文件夹"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_imgs.clicked.connect(self.button_images_open)
        self.pushButton_imgfile.clicked.connect(self.button_imagefile_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.pushButton_showdir.clicked.connect(self.button_show_dir)
        self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('')   # 绘制初始化图片
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    # 退出提示
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def button_image_open(self):
        print('打开图片')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            self.empty_information()
            print('empty!')
            return
        img = cv2.imread(img_name)
        print(img_name)
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.imgsz)[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], showimg.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        # print(label.split()[0])  # 打印各目标名称
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=label,
                                     color=self.colors[int(cls)], line_thickness=2)

        cv2.imwrite(str(Path(self.save_file / 'prediction.jpg')), showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        print('单图片检测完成')

    def button_images_open(self):
        print('打开图片')

        img_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if len(img_names) == 0:
            self.empty_information()
            print('empty!')
            return
        index = 0
        for img_name in img_names:
            name_list = []
            img = cv2.imread(img_name)
            print(img_name)
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            # print(label.split()[0])  # 打印各目标名称
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

            cv2.imwrite(str(Path(self.save_file / 'prediction_imgs{}.jpg'.format(index))), showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            index += 1
        print('多图片检测完成')

    def button_imagefile_open(self):
        print('打开图片文件夹')

        file_name = QtWidgets.QFileDialog.getExistingDirectory(
            self, "打开图片文件夹", "")
        if not file_name:
            self.empty_information()
            print('empty!')
            return
        print(file_name)
        img_names = os.listdir(file_name)
        if len(img_names) == 0:
            self.empty_information()
            print('empty!')
            return
        index = 0
        for img_name in img_names:
            if img_name.split('.')[-1] not in ('jpg', 'png', 'jpeg'):
                continue
            name_list = []
            img = cv2.imread(os.path.join(file_name, img_name))
            print(img_name)
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            # print(label.split()[0])  # 打印各目标名称
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

            cv2.imwrite(str(Path(self.save_file / 'prediction_file{}.jpg'.format(index))), showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                      QtGui.QImage.Format_RGB32)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            index += 1
        print('文件夹图片检测完成')

    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")

        if not video_name:
            self.empty_information()
            print('empty!')
            return

        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter(str(Path(self.save_file / 'vedio_prediction.avi')), cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)

    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter(str(Path(self.save_file / 'camera_prediction.avi')), cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头检测")

    def show_video_frame(self):
        name_list = []

        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            # print(label)  # 打印各目标+置信度
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.init_logo()

    def button_show_dir(self):
        path = self.save_file
        os.system(f"start explorer {path}")

    def empty_information(self):
        QtWidgets.QMessageBox.information(self, '提示', '未选择文件或选择文件为空!', QtWidgets.QMessageBox.Cancel)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ui = Ui_MainWindow()
    # 设置窗口透明度
    ui.setWindowOpacity(0.93)
    # 去除顶部边框
    # ui.setWindowFlags(Qt.FramelessWindowHint)
    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon.ico"), QIcon.Normal, QIcon.Off)
    # 设置应用图标
    ui.setWindowIcon(icon)
    ui.show()
    sys.exit(app.exec_())

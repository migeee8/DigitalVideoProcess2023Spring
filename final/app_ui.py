"""
GUI实现

Ui_MainWindow类实现了应用的用户交互界面：
    - 用户从菜单中选择需要识别的图片或者视频进行识别
    - 识别的信息会在下方图片或者视频中显示
    - 最下方输出识别车牌信息
"""

from yolov7_detect_rec import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1350, 774)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.videoArea = QtWidgets.QLabel(self.centralwidget)
        self.videoArea.setGeometry(QtCore.QRect(360, 20, 941, 521))
        self.videoArea.setFrameShape(QtWidgets.QFrame.Box)
        self.videoArea.setText("")
        self.videoArea.setPixmap(QtGui.QPixmap("init.png"))
        self.videoArea.setScaledContents(True)
        self.videoArea.setObjectName("videoArea")
        self.message = QtWidgets.QLabel(self.centralwidget)
        self.message.setGeometry(QtCore.QRect(200, 570, 941, 121))
        self.message.setMaximumSize(QtCore.QSize(16777205, 16777215))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(18)
        self.message.setFont(font)
        self.message.setFrameShape(QtWidgets.QFrame.Box)
        self.message.setFrameShadow(QtWidgets.QFrame.Plain)
        self.message.setLineWidth(2)
        self.message.setScaledContents(False)
        self.message.setAlignment(QtCore.Qt.AlignCenter)
        self.message.setObjectName("message")
        self.title_label2 = QtWidgets.QLabel(self.centralwidget)
        self.title_label2.setGeometry(QtCore.QRect(50, 590, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(16)
        self.title_label2.setFont(font)
        self.title_label2.setObjectName("title_label2")
        self.keyframe = QtWidgets.QLabel(self.centralwidget)
        self.keyframe.setGeometry(QtCore.QRect(30, 100, 301, 181))
        self.keyframe.setFrameShape(QtWidgets.QFrame.Box)
        self.keyframe.setText("")
        self.keyframe.setScaledContents(True)
        self.keyframe.setObjectName("keyframe")
        self.plateImg = QtWidgets.QLabel(self.centralwidget)
        self.plateImg.setGeometry(QtCore.QRect(60, 350, 241, 101))
        self.plateImg.setFrameShape(QtWidgets.QFrame.Box)
        self.plateImg.setText("")
        self.plateImg.setScaledContents(True)
        self.plateImg.setObjectName("plateImg")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1350, 26))
        self.menubar.setObjectName("menubar")
        self.menuFIle = QtWidgets.QMenu(self.menubar)
        self.menuFIle.setObjectName("menuFIle")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionImage = QtWidgets.QAction(MainWindow)
        self.actionImage.setObjectName("actionImage")
        self.actionVideo = QtWidgets.QAction(MainWindow)
        self.actionVideo.setObjectName("actionVideo")
        self.menuFIle.addAction(self.actionImage)
        self.menuFIle.addAction(self.actionVideo)
        self.menubar.addAction(self.menuFIle.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.actionImage.triggered.connect(self.load_image_action)
        self.actionVideo.triggered.connect(self.load_video_action)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "License Plate Recognition"))
        self.message.setText(_translate("MainWindow", " "))
        self.title_label2.setText(_translate("MainWindow", "车牌信息"))
        self.menuFIle.setTitle(_translate("MainWindow", "打开文件"))
        self.actionImage.setText(_translate("MainWindow", "Image"))
        self.actionVideo.setText(_translate("MainWindow", "Video"))

    def load_image_action(self):
        # 清空
        pixmap = QtGui.QPixmap()
        self.plateImg.setPixmap(pixmap)
        self.keyframe.setPixmap(pixmap)
        self.message.setText(" ")

        # 数据准备
        file_th, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                           "Image Files (*.png *.jpg *.jpeg *.bmp)")
        detect_model = 'weights/yolov7-lite-s.pt'  # 目标检测模型
        device = ("cuda" if torch.cuda.is_available() else "cpu")  # 设备
        model = attempt_load(detect_model, map_location=device)  # 载入目标检测模型
        rec_model = 'weights/plate_rec_color.pth'  # 识别模型
        output = 'result'  # 输出路径
        source = file_th  # 源路径
        img_size = 640
        plate_rec_model = init_model(device, rec_model)  # 载入识别模型

        if not os.path.exists(output):
            os.mkdir(output)  # 如果输出路径文件夹不存在 新建文件夹
        time_b = time.time()
        # print(file_th, end=" ")

        img = cv_imread(file_th)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 获得车牌目标检测识别的信息
        dict_list = detect_Recognition_plate(model, img, device, plate_rec_model, img_size)
        for result in dict_list:
            self.message.setText(result['plate_no'])  # 显示检测的车牌号
            coordinate = result['landmarks']

        # 绘制识别结果到原始图像上，并在矩形框内添加车牌号和颜色 并保存结果图片
        res_img = draw_result(img, dict_list)
        img_name = os.path.basename(file_th)
        save_img_path = os.path.join(output, img_name)
        cv2.imwrite(save_img_path, res_img)

        img_copy = copy.deepcopy(res_img)
        plate_img = crop_plate_from_image(img_copy, coordinate)
        # # cv2.imshow("plate", plate_img)
        height_p, width_p, channel_p = plate_img.shape
        bytesPerLine_p = 3 * width_p
        plate_img_bytes = plate_img.tobytes()
        qImg_plate = QtGui.QImage(plate_img_bytes, width_p, height_p, bytesPerLine_p, QtGui.QImage.Format_RGB888)
        platePix = QtGui.QPixmap.fromImage(qImg_plate)
        self.plateImg.setPixmap(platePix.scaled(self.plateImg.size(), QtCore.Qt.KeepAspectRatio))

        # 设置为pixmap格式 更新label框显示
        pixmap = QPixmap(save_img_path)
        self.videoArea.setPixmap(pixmap.scaled(self.videoArea.size(), QtCore.Qt.KeepAspectRatio))
        self.keyframe.setPixmap(pixmap.scaled(self.keyframe.size(), QtCore.Qt.KeepAspectRatio))


    def load_video_action(self):
        pixmap = QtGui.QPixmap()
        self.plateImg.setPixmap(pixmap)
        self.keyframe.setPixmap(pixmap)
        self.message.setText(" ")

        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Video", "",
                                                              "Video Files (*.mp4 *.avi *.mov)")
        capture = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = capture.get(cv2.CAP_PROP_FPS)  # 帧数
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高

        frame_count = 0
        fps_all = 0
        # rate,FrameNumber,duration=get_second(capture)
        img_size = 640
        detect_model = 'weights/yolov7-lite-s.pt'
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model = attempt_load(detect_model, map_location=device)
        rec_model = 'weights/plate_rec_color.pth'
        plate_rec_model = init_model(device, rec_model)

        keyframe = 0;

        while capture.isOpened():
            t1 = cv2.getTickCount()  # 获取时间戳
            frame_count += 1
            print(f"第{frame_count} 帧", end=" ")

            ret, img = capture.read()
            if not ret:
                break
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            dict_list = detect_Recognition_plate(model, img, device, plate_rec_model, img_size)
            for result in dict_list:
                self.message.setText(result['plate_no'])
                coordinate = result['landmarks']
            ori_img = draw_result(img, dict_list)

            t2 = cv2.getTickCount()
            infer_time = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / infer_time
            fps_all += fps

            rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

            if len(dict_list) > 0 and keyframe == 0:
                height, width, channel = rgb_img.shape
                bytesPerLine = 3 * width
                qImg = QtGui.QImage(rgb_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                framePix = QtGui.QPixmap.fromImage(qImg)
                self.keyframe.setPixmap(framePix.scaled(self.keyframe.size(), QtCore.Qt.KeepAspectRatio))

                img_copy = copy.deepcopy(img)
                plate_img = crop_plate_from_image(img_copy, coordinate)
                # # cv2.imshow("plate", plate_img)
                height_p, width_p, channel_p = plate_img.shape
                bytesPerLine_p = 3 * width_p
                plate_img_bytes = plate_img.tobytes()
                qImg_plate = QtGui.QImage(plate_img_bytes, width_p, height_p, bytesPerLine_p, QtGui.QImage.Format_RGB888)
                platePix = QtGui.QPixmap.fromImage(qImg_plate)
                self.plateImg.setPixmap(platePix.scaled(self.plateImg.size(), QtCore.Qt.KeepAspectRatio))

                keyframe = 1

            picture = QtGui.QImage(rgb_img, width, height, 3 * width, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(picture)
            self.videoArea.setPixmap(pixmap.scaled(self.videoArea.size(), QtCore.Qt.KeepAspectRatio))
            cv2.waitKey(10)

        capture.release()
        # out.release()
        print(f"all frame is {frame_count},average fps is {fps_all / frame_count} fps")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

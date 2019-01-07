from __future__ import division

import cv2
import numpy as np
import glob
import os
import re
import sys
from copy import deepcopy
from collections import namedtuple

from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import ImageProcessingLibrary
from dialog import Ui_dialog

CB_VALUE = 255
CR_VALUE = 255
LIMIT_VALUE = 255

TIMER_INTERVAL = 500
SLIDESHOW_STEP = 0
SLIDESHOW_INTERVAL = 300  # milliseconds


class MyForm(QDialog):
    HSVColorFilter = ImageProcessingLibrary.HSVColorFilter()
    Segmentation = ImageProcessingLibrary.Segmentation()
    OpticalFlow = ImageProcessingLibrary.Optical_Flow()

    def __init__(self):
        super().__init__()
        self.ready = False
        self.ui = Ui_dialog()
        self.ui.setupUi(self)
        self.threadpool = QThreadPool()
        self.show()

        ###########################################################
        # Use QSettings to save states
        ###########################################################

        self.settings = QSettings('__settings.ini', QSettings.IniFormat)
        self.settings.setFallbacksEnabled(False)

        ###########################################################
        # Connect Signal
        ###########################################################
        # self.ui.label_finalImage.mousePressEvent = self.getPos
        self.ui.label_finalImage.installEventFilter(self)
        self.ui.label_finalImage.setMouseTracking(True)
        # self.ui.label_initImage.linkHovered.connect(self.leftHovered)
        # self.ui.label_finalImage.linkHovered.connect(self.rightHovered)

        ####################################################################
        # Slider Signals
        ####################################################################
        self.ui.horizontalSlider_cb.valueChanged.connect(self.updateSlider)
        self.ui.horizontalSlider_cr.valueChanged.connect(self.updateSlider)
        self.ui.horizontalSlider_limit.valueChanged.connect(self.updateSlider)

        ####################################################################
        # SpinBox Signals
        ####################################################################
        self.ui.spinBox_cb.valueChanged.connect(self.updateSpinboxValue)
        self.ui.spinBox_cr.valueChanged.connect(self.updateSpinboxValue)
        self.ui.spinBox_limit.valueChanged.connect(self.updateSpinboxValue)

        ####################################################################
        # Buttons Signals
        ####################################################################
        self.ui.pushButton_Step_Left.clicked.connect(self.backward_oneImage)
        self.ui.pushButton_Step_Right.clicked.connect(self.forward_oneImage)
        self.ui.pushButton_browse.clicked.connect(self.openImageSeq)
        self.ui.pushButton_RunImgSeq.clicked.connect(self.startStopSlideShow)
        self.ui.pushButton_close.clicked.connect(QCoreApplication.instance().quit)
        self.ui.pushButton_close.clicked.connect(self.close)

        ####################################################################
        # Combobox Signals
        ####################################################################
        self.ui.cb_segment_method.currentIndexChanged.connect(self.cb_selectionChanged)
        self.ui.cb_segment_method.currentIndexChanged.connect(self.request_image_update)

        self.ui.chkBox_with_flow.clicked.connect(self.request_image_update)

        ###########################################################
        # Labels
        ###########################################################
        self.ui.lbl_info.setText('')

        ###########################################################
        # Initial value
        ###########################################################
        # Init slider value for high Threshold
        self.ui.horizontalSlider_cb.setValue(CB_VALUE)
        self.ui.horizontalSlider_cr.setValue(CR_VALUE)
        self.ui.horizontalSlider_limit.setValue(LIMIT_VALUE)
        self.ui.spinBox_cb.setValue(CB_VALUE)
        self.ui.spinBox_cr.setValue(CR_VALUE)
        self.ui.spinBox_limit.setValue(LIMIT_VALUE)
        # Set scaled properties
        self.ui.label_initImage.setScaledContents(True)
        self.ui.label_finalImage.setScaledContents(True)

        ###########################################################
        # Init Combo Box
        ###########################################################
        self.ui.cb_segment_method.addItems(
            ["Choose Method:", "YCbCr Background Subtractor", "Contour"])
        self.ui.cb_segment_method.setCurrentIndex(0)

        ###########################################################
        # Timer update slider values
        ###########################################################
        self.timer_update = QTimer()
        self.timer_update.timeout.connect(self.updateResultImage)
        self.timer_update.start(TIMER_INTERVAL)

        ###########################################################
        # Timer for image slide show
        ###########################################################
        self.timer_slideshow = QTimer()
        self.timer_slideshow.timeout.connect(self.runSlideShow)
        # self.timer_slideshow.setInterval(SLIDESHOW_DELAY)
        self.slideshow_step = SLIDESHOW_STEP

        ###########################################################
        #  Lists concerning images
        ###########################################################
        self.image_path_list = []
        self.openCVImg_initial_img_list = []
        self.openCVImg_final_img_list = []
        self.pixMapImg_initial_img_list = []  # shown in left label
        self.pixMapImg_initial_flow_list = []
        self.pixMapImg_original_list = []  # unaltered pixMap images
        self.pixMapImg_final_img_list = []  # shown in right label

        ###########################################################
        #  Misc Variables
        ###########################################################
        self.tot_numb_of_images = 0
        self.name_of_current_imgProcFunc = None  # name (string) of currently used img proc function
        self.image_mask = None
        self.imageLoaded = False
        self.isImageValid = False
        self.scale_fac_width  = None
        self.scale_fac_height = None
        self.qimage_width  = 2592
        self.qimage_height = 1944
        self.qLable_width = None
        self.qLable_height = None
        self.pass_this_imgProcFunc = None  # Placeholder for currently used img proc function
        self.delta = 15  # number of imgs to be let out before processing starts

        ###########################################################
        #  Boolean variables
        ###########################################################
        self.optFlowList_exists = False  # If a list of optical flow img's exists

        ###########################################################
        # Load images from last session as background task
        ###########################################################
        self.path_to_images = None
        self.path_to_images = self.settings.value("path_to_images")
        temp_text = self.prepare_last_image_selection(self.path_to_images)
        self.ui.lineEdit.setText(temp_text)
        self.ui.lbl_pixel_coordinatesAndValue.setText('')

        ###########################################################
        # Named Tuples used as Structures
        ###########################################################
        #self.RGB = namedtuple("RGB", "red green blue")
        #self.RGB(red = 0, green=0, blue= 0)

        ###########################################################
        # All variables declared and ready to be used
        ###########################################################
        self.ready = True  # If iInitialization completed

    #########################################################
    # File Dialoge
    #########################################################
    def prepare_last_image_selection(self, last_path):

        if not not last_path:
            last_path = re.sub(r'\s+', '', last_path)
            self.path_to_images = last_path

            for file in sorted(glob.glob(os.path.join(str(last_path), '*.jpg'))):
                self.image_path_list.append(file)

            if len(self.image_path_list) > 0:
                # worker = Worker(self.preprocessImages)
                # self.threadpool.start(worker)

                pool = ThreadPoolExecutor(max_workers=3)
                pool.submit(self.prepareImages)

        return last_path

    def prepareImages(self):

        try:
            self.tot_numb_of_images = len(self.image_path_list)
            num_of_preloaded_imgs = self.tot_numb_of_images // 2
            img_cnt = 0
            ready = True

            self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: #f0f0f0")
            self.ui.pushButton_browse.setEnabled(False)
            self.ui.pushButton_RunImgSeq.setEnabled(False)
            self.ui.lbl_progress_status.setText('Loading Images')
            status_txt = 'Loading Images . . .'

            for new_path in self.image_path_list:
                opencv_img = cv2.imread(new_path)
                opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

                self.openCVImg_initial_img_list.append(deepcopy(opencv_img))
                self.openCVImg_final_img_list.append(deepcopy(opencv_img))
                img_cnt += 1

                masked_OpenCV_Image = self.Segmentation.maske_OpenCV_Image(opencv_img)

                self.pixMapImg_initial_img_list.append(self.cv2qpixmap(deepcopy(masked_OpenCV_Image)))
                self.pixMapImg_final_img_list.append(self.cv2qpixmap(deepcopy(masked_OpenCV_Image)))
                self.pixMapImg_original_list.append(self.cv2qpixmap(deepcopy(masked_OpenCV_Image)))


                status_txt = status_txt + ' .'
                self.ui.lbl_progress_status.setText(status_txt)

                if (img_cnt >= num_of_preloaded_imgs) & ready:
                    self.ui.lbl_progress_status.setText('')
                    self.ui.pushButton_RunImgSeq.setText('Ready')
                    self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: yellow")
                    self.ui.pushButton_RunImgSeq.setEnabled(True)
                    self.ui.pushButton_browse.setEnabled(True)
                    self.ui.chkBox_with_flow.setEnabled(True)
                    ready = False

            self.ui.lbl_progress_status.setText('Done')

        except Exception as e:
            print("Error: could not load image: ", str(e))

    def getImageDirectory(self):
        pathString = self.image_path_list[0]
        filePath = pathString.split('/')
        filePath = filePath[:-1]
        imageDirectory = ''
        for item in filePath:
            imageDirectory = os.path.join(imageDirectory, item)

        bs = r'\ '

        imageDirectory = imageDirectory.replace(':', ':' + bs).strip()
        imageDirectory = re.sub(r'\s+', '', imageDirectory)
        return imageDirectory

    def openImageSeq(self):
        try:

            if not not self.path_to_images:
                open_directory = self.path_to_images
            else:
                open_directory = os.getcwd()
                self.ui.lineEdit.setText(open_directory)

            self.image_path_list, _ = QFileDialog.getOpenFileNames(
                caption="Select one or more files to open",
                directory=open_directory,
                filter="*.jpg *.png")
            if self.image_path_list:
                self.isImageValid = True
                self.ui.lineEdit.setText(self.getImageDirectory())
                self.prepareImages()

            else:
                self.isImageValid = False
        except Exception as e:
            QMessageBox.about(self, "Error: could not read image path: ", str(e))

    def request_image_update(self, state):

        if not self.ready:
            return

        _sender = self.sender()
        _sender_name = _sender.objectName()

        print('Sender Name {} '.format(_sender_name))

        if _sender_name == 'cb_segment_method':
            #print('ComboBox: {}'.format(_sender.currentText()))

            if self.name_of_current_imgProcFunc == "Choose Method:":
                self.pixMapImg_final_img_list = self.pixMapImg_original_list[:]
                return

            if self.name_of_current_imgProcFunc == "YCbCr Background Subtractor":
                self.pass_this_imgProcFunc = self.Segmentation.ycbcr_substractor

                pool = ThreadPoolExecutor(max_workers=4)
                pool.submit(self.doImageProcessing_1)
                return

            if self.name_of_current_imgProcFunc == "Contour":
                # print('----> Selected segmentation Method: {}'.format(_sender.currentText()))
                self.pass_this_imgProcFunc = self.Segmentation.findContour

                pool = ThreadPoolExecutor(max_workers=4)
                pool.submit(self.doImageProcessing_1)
                return

            if self.name_of_current_imgProcFunc == "Blue Red Ratio":
                # print('----> Selected segmentation Method: {}'.format(_sender.currentText()))
                return

            if self.name_of_current_imgProcFunc == "Method 3":
                # print('----> Selected segmentation Method: {}'.format(_sender.currentText()))
                return

        if _sender_name == 'chkBox_with_flow':

            if _sender.isChecked():
                self.name_of_current_imgProcFunc = 'Optical Flow'
                self.pass_this_imgProcFunc = self.OpticalFlow.opt_flow
                pool = ThreadPoolExecutor(max_workers=4)
                pool.submit(self.doImageProcessing_2)
            else:
                self.name_of_current_imgProcFunc = "None"
                self.pixMapImg_initial_img_list = self.pixMapImg_original_list[:]

    #########################################################
    # Image Processing
    #########################################################

    def doImageProcessing_1(self):

        try:
            imgProcFunc = self.pass_this_imgProcFunc
            delta = self.delta

            whereException = ''
            cur_idx = self.slideshow_step
            last_idx = self.tot_numb_of_images

            if cur_idx + delta > last_idx:
                start_1 = (cur_idx + delta) - last_idx
                start_2 = 0
                stop_1 = last_idx
                stop_2 = start_1
                whereException = 'I | '

            elif cur_idx + delta < last_idx:
                start_1 = (cur_idx + delta)
                start_2 = 0
                stop_1 = last_idx
                stop_2 = start_1

            elif cur_idx + delta == last_idx:
                start_1 = 0

                for index in range(last_idx):
                    image = deepcopy(self.openCVImg_final_img_list[index])
                    self.pixMapImg_final_img_list[index] = self.cv2qpixmap(imgProcFunc(image))
                    whereException = 'I '
                    self.ui.lbl_info.setText('Current: {} | preproc {}'.format(self.slideshow_step, index))
                return

            for index in range(start_1, stop_1):
                image = deepcopy(self.openCVImg_final_img_list[index])
                self.pixMapImg_final_img_list[index] = self.cv2qpixmap(imgProcFunc(image))
                whereException = 'II '
                self.ui.lbl_info.setText('Current: {} | preproc {}'.format(self.slideshow_step, index))

            for index in range(start_2, stop_2):
                image = deepcopy(self.openCVImg_final_img_list[index])
                self.pixMapImg_final_img_list[index] = self.cv2qpixmap(imgProcFunc(image))
                whereException = 'III '
                self.ui.lbl_info.setText('Current: {} | preproc {}'.format(self.slideshow_step, index))

            self.ui.lbl_info.setText('')

        except Exception as e:
            print('Where : {}{}'.format(whereException, str(e)))

    def doImageProcessing_2(self):
        try:
            # Do not reprocess imgs if list of optical flow already exists
            if self.check_if_list_exists():
                return
            print('Recalculating  OptFlow')

            imgProcFunc = self.pass_this_imgProcFunc
            delta = self.delta

            whereException = ''
            cur_idx = self.slideshow_step
            last_idx = self.tot_numb_of_images

            if cur_idx + delta > last_idx:
                start_1 = (cur_idx + delta) - last_idx
                start_2 = 0
                stop_1 = last_idx
                stop_2 = start_1

            elif cur_idx + delta < last_idx:
                start_1 = (cur_idx + delta)
                start_2 = 0
                stop_1 = last_idx
                stop_2 = start_1

            elif cur_idx + delta == last_idx:

                for index in range(last_idx):

                    if index == 0:
                        idx_prev = last_idx - 1
                        idx_next = index
                    else:
                        idx_prev = index - 1
                        idx_next = index

                    whereException = 'I '
                    prev = deepcopy(self.openCVImg_initial_img_list[idx_prev])
                    next = deepcopy(self.openCVImg_initial_img_list[idx_next])

                    flow_img = self.cv2qpixmap(imgProcFunc(prev, next))
                    self.pixMapImg_initial_img_list[index] = flow_img
                    self.pixMapImg_initial_flow_list.append(flow_img)

                    self.ui.lbl_info.setText('Current: {} | preproc {}'.format(self.slideshow_step, index))

                self.set_new_imgList_exists()
                self.ui.lbl_info.setText('')
                return

            for index in range(start_1, stop_1):

                if index == 0:
                    idx_prev = last_idx - 1
                    idx_next = index
                else:
                    idx_prev = index - 1
                    idx_next = index

                whereException = 'II '
                prev = deepcopy(self.openCVImg_initial_img_list[idx_prev])
                next = deepcopy(self.openCVImg_initial_img_list[idx_next])

                flow_img = self.cv2qpixmap(imgProcFunc(prev, next))
                self.pixMapImg_initial_img_list[index] = flow_img
                self.pixMapImg_initial_flow_list.append(flow_img)

                self.ui.lbl_info.setText('Current: {} | preproc {}'.format(self.slideshow_step, index))

            for index in range(start_2, stop_2):
                if index == 0:
                    idx_prev = last_idx - 1
                    idx_next = index
                else:
                    idx_prev = index - 1
                    idx_next = index

                whereException = 'III '
                prev = deepcopy(self.openCVImg_initial_img_list[idx_prev])
                next = deepcopy(self.openCVImg_initial_img_list[idx_next])

                flow_img = self.cv2qpixmap(imgProcFunc(prev, next))
                self.pixMapImg_initial_img_list[index] = flow_img
                self.pixMapImg_initial_flow_list.append(flow_img)

                self.ui.lbl_info.setText('Current: {} | preproc {}'.format(self.slideshow_step, index))

            self.set_new_imgList_exists()
            self.ui.lbl_info.setText('')

        except Exception as e:
            print('Where : {}{}'.format(whereException, str(e)))

    def set_new_imgList_exists(self):

        if self.name_of_current_imgProcFunc == 'Optical Flow':
            self.optFlowList_exists = True

    def check_if_list_exists(self):

        if (self.name_of_current_imgProcFunc == 'Optical Flow') & (self.optFlowList_exists):
            self.pixMapImg_initial_img_list = self.pixMapImg_initial_flow_list[:]
            return True
        else:
            False

    def showFilteredImage(self):
        if self.isImageValid:
            # Convert BGR to RGB
            self.cvFinalImage = self.HSVColorFilter.getFilteredImage()
            # print self.cvOriginalImage
            self.cvFinalImage = cv2.cvtColor(self.cvFinalImage, cv2.COLOR_HSV2RGB)
            # Get image properties
            height, width, byteValue = self.cvFinalImage.shape

            # Convert to QPixmap
            self.qImageData = QImage(self.cvFinalImage, width, height, QImage.Format_RGB888)
            self.qPixmapData = QPixmap.fromImage(self.qImageData)

            # scale and place on label
            self.ui.label_finalImage.setPixmap(self.qPixmapData)

    #########################################################
    # Selecting Methodes
    #########################################################
    def cb_selectionChanged(self):
        current_selection = self.ui.cb_segment_method.currentText()
        self.name_of_current_imgProcFunc = current_selection

    #########################################################
    # Navigating through images
    #########################################################
    def runSlideShow(self):

        try:
            if self.slideshow_step >= self.tot_numb_of_images:
                self.slideshow_step = 0

            self.ui.label_initImage.setPixmap(self.pixMapImg_initial_img_list[self.slideshow_step])
            self.ui.label_finalImage.setPixmap(self.pixMapImg_final_img_list[self.slideshow_step])

            self.slideshow_step += 1

            self.ui.lbl_progress_status.setText(
                'Showing image {} of {} '.format(self.slideshow_step, self.tot_numb_of_images))

        except Exception as e:
            print('Backward one image: {}'.format(e))

    def startStopSlideShow(self):

        if len(self.image_path_list) == 0:
            return

        btn_text = self.ui.pushButton_RunImgSeq.text()

        if btn_text == 'Ready':
            self.imageLoaded = True
            self.ui.pushButton_RunImgSeq.setText('Stop')
            self.timer_slideshow.start(SLIDESHOW_INTERVAL)
            self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: red")
            return

        if btn_text == 'Run':
            self.ui.pushButton_RunImgSeq.setText('Stop')
            self.timer_slideshow.start(SLIDESHOW_INTERVAL)
            self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: red")
            self.ui.pushButton_Step_Right.setEnabled(False)
            self.ui.pushButton_Step_Left.setEnabled(False)
            return

        if btn_text == 'Stop':
            self.ui.pushButton_RunImgSeq.setText('Run')
            self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: green")
            self.timer_slideshow.stop()
            self.ui.pushButton_Step_Right.setEnabled(True)
            self.ui.pushButton_Step_Left.setEnabled(True)

    def backward_oneImage(self):
        try:
            self.slideshow_step -= 1

            if self.slideshow_step < 0:
                self.slideshow_step = self.tot_numb_of_images - 1
                current_image = self.slideshow_step
            if self.slideshow_step == 0:
                current_image = self.tot_numb_of_images
            else:
                current_image = self.slideshow_step

            self.ui.label_initImage.setPixmap(self.pixMapImg_initial_img_list[self.slideshow_step])
            self.ui.label_finalImage.setPixmap(self.pixMapImg_final_img_list[self.slideshow_step])

            self.ui.lbl_progress_status.setText(
                'Showing image {} of {} '.format(current_image, self.tot_numb_of_images))

        except Exception as e:
            print('Slideshow : {}'.format(e))

    def forward_oneImage(self):

        try:
            self.slideshow_step += 1

            if self.slideshow_step >= self.tot_numb_of_images:
                self.slideshow_step = 0
                current_image = self.tot_numb_of_images
            else:
                current_image = self.slideshow_step

            self.ui.label_initImage.setPixmap(self.pixMapImg_initial_img_list[self.slideshow_step])
            self.ui.label_finalImage.setPixmap(self.pixMapImg_final_img_list[self.slideshow_step])

            self.ui.lbl_progress_status.setText(
                'Showing image {} of {} '.format(current_image, self.tot_numb_of_images))

        except Exception as e:
            print('Foreward one image: {}'.format(e))

    #########################################################
    # Misc
    #########################################################
    def updateScaleFac(self):
        self.scale_fac_width  = np.rint(self.qimage_width  / self.qLable_width)
        self.scale_fac_height = np.rint(self.qimage_height / self.qLable_height)

    def calcTruePixCoordinates(self, mouse_x, mouse_y):
        true_x = mouse_x * self.scale_fac_width
        if true_x > self.qimage_width:
            true_x = self.qimage_width

        true_y = mouse_y * self.scale_fac_height
        if true_y > self.qimage_height:
            true_y = self.qimage_height

        trueCoordinates = QPoint(true_x,true_y)

        return trueCoordinates

    def getPixelValue(self,true_x,true_y):

        try:
            if self.imageLoaded:
                if len(self.pixMapImg_final_img_list) > 0:

                    qimage = self.pixMapImg_final_img_list[self.slideshow_step - 1].toImage()
                    pixel_val = qimage.pixel(true_x, true_y)

                    _red   = qRed(pixel_val)
                    _green = qGreen(pixel_val)
                    _blue  = qBlue(pixel_val)

                    RGB = namedtuple("RGB", "red green blue")
                    pixel_rgb = RGB(red=_red, green=_green, blue=_blue)

                return pixel_rgb

        except Exception as e:
            print('Error getPixelValue: {}'.format(e))

    def updatePixelText(self,x,y):

        if self.imageLoaded:
            pixel_coords = self.calcTruePixCoordinates(x, y)
            true_x = pixel_coords.x()
            true_y = pixel_coords.y()
            pixel_rgb = self.getPixelValue(true_x, true_y)
            pixel_coord_value = "Pixel x,y: ({},{}) ->  [{},{},{}]" \
                .format(true_x, true_y, pixel_rgb.red, pixel_rgb.green, pixel_rgb.blue)
            self.ui.lbl_pixel_coordinatesAndValue.setText(pixel_coord_value)

    def closeEvent(self, event):

        if self.path_to_images is not None:
            self.settings.setValue("path_to_images", self.path_to_images)
            event.accept()

    def cv2qpixmap(self, openCV_img):
        height, width, channel = openCV_img.shape
        qt_img = QImage(openCV_img, width, height, QImage.Format_RGB888)
        qpixmap_img = QPixmap.fromImage(qt_img)
        return qpixmap_img

    ########################################################
    # QSlider
    ########################################################
    def updateResultImage(self):
        low_Hue = self.ui.horizontalSlider_cb.value()
        low_Sat = self.ui.horizontalSlider_cr.value()
        low_Val = self.ui.horizontalSlider_limit.value()


        # Please note that we send value in RGB format

        # Hoa für HSF FILTER wieder einkommentieren !

        # But openCV process in GRB. Additional processing is handled in ImageProcessingLibrary class
        # self.HSVColorFilter.filterImage((low_Hue, low_Sat, low_Val), (high_Hue, high_Sat, high_Val))
        # self.showFilteredImage()

    def updateSlider(self):
        self.ui.spinBox_cb.setValue(self.ui.horizontalSlider_cb.value())
        self.ui.spinBox_cr.setValue(self.ui.horizontalSlider_cr.value())
        self.ui.spinBox_limit.setValue(self.ui.horizontalSlider_limit.value())

    def updateSpinboxValue(self):
        self.ui.horizontalSlider_cb.setValue(self.ui.spinBox_cb.value())
        self.ui.horizontalSlider_cr.setValue(self.ui.spinBox_cr.value())
        self.ui.horizontalSlider_limit.setValue(self.ui.spinBox_limit.value())

    #########################################################
    # Mouse
    #########################################################

    def draw_circle(self, event, x, y, flags, param):
        global ix, iy
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
            self.mouseX, self.mouseY = x, y

    def eventFilter(self, srcEvent, event):
        try:

            x = 0
            y = 0

            if srcEvent == self.ui.label_finalImage:
                if event.type() == QEvent.Leave:
                    self.ui.lbl_pixel_coordinatesAndValue.setText('')

                if event.type() == QEvent.Resize:
                    self.qLable_width  = srcEvent.width()
                    self.qLable_height = srcEvent.height()

                    self.updateScaleFac()
                    #print('scaleFactor : width: {}  height: {}'.format(self.scale_fac_width,self.scale_fac_height))

                if event.type() == QEvent.MouseMove:
                    x = event.pos().x()
                    y = event.pos().y()
                    self.updatePixelText(x,y)

                elif event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.RightButton:
                        # print('Rightclick')
                        return False

                    elif event.button() == Qt.LeftButton:
                        # print('Leftclick ')

                        x = event.pos().x()
                        y = event.pos().y()
                        self.updatePixelText(x, y)

                        return False
                    else:
                        return False

            return False


        except Exception as e:
            print('Error evenFilter: {}'.format(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())

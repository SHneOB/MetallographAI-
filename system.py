from PyQt5.QtCore import QUrl, QSize, QFileInfo
from PyQt5.QtWidgets import QWidget, QMessageBox, QTableWidgetItem, QHeaderView, QTableWidget, \
    QFileDialog, QTreeWidgetItem, QAbstractItemView, QApplication, QFileIconProvider
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5 import QtGui, Qt

from ui.mainWindow import Ui_Form_main_window

from utils import *
#import subprocess 
#from tensorboardX import SummaryWriter  
from tensorboard import program
from train import TrainThread
from detect import ClassifyThread
import torch
import os
import shutil

class System:
    def __init__(self, db_config: dict = None):
        super().__init__()

        # Init login and main window
        self.mw_widget = QWidget()

        self.mw_ui = Ui_Form_main_window()#界面
        self.mw_ui.setupUi(self.mw_widget)

        self.tb = None
        self.tb_pid = 0
        self.log_path = 'runs/classify'
        self.lastfile_path = './'
        self.model_name = None
        self.cls_result = None
        self.train_result = None
        self.cls_input_image_path = None
        self.exist_model_save_name = []
        self.usr_authority = True

        self.web_view = None

        self.model_config = {
            'weight': 'None',
            'epochs': 0,
            'batch_size': 0,
            'lr': 0.0,
            'optimizer': 'None',
            'device': 'None',
            'weight_path': 'weights',
            'imgsz': 0,
            'workers': 0,
            'dataset': 'None'
        }

        self.__mw_init()#调用main window初始化

    def __mw_init(self):
        # Init Model train
        if not torch.cuda.is_available():
            self.mw_ui.comboBox_device.removeItem(0)
        self.mw_ui.pushButton_startTrain.clicked.connect(self.__train_model)
        self.__visualizer_init()

        # Init Model detect
        self.__update_comboBox_model()
        self.mw_ui.pushButton_selectPic.clicked.connect(
            lambda: self.__get_picture_from_file_dialog(self.mw_ui.input_pic_widget))
        self.mw_ui.pushButton_startDetect.clicked.connect(self.__begin_classify)

        # Init Dataset management
        self.mw_ui.pushButton_upload.clicked.connect(
            lambda: self.__get_picture_from_file_dialog(self.mw_ui.preview_pic_widget))
        self.mw_ui.pushButton_delpic.clicked.connect(self.__delpic)
        self.mw_ui.pushButton_calibrate.clicked.connect(self.__calibrate)
        self.mw_ui.pushButton_calibrate_val.clicked.connect(self.__calibrate_val)
        self.mw_ui.pushButton_calibrate_test.clicked.connect(self.__calibrate_test)
        self.__filetree_init()
    #calibrate文件夹
    def __filetree_init(self):
        self.__get_model_config()
        path = f"datasets/{self.model_config['dataset']}"#加上train则只显示目标数据集中的train文件

        self.mw_ui.treeWidget.clear()

        self.mw_ui.treeWidget.setColumnCount(1)
        self.mw_ui.treeWidget.setColumnWidth(0, 50)
        self.mw_ui.treeWidget.setHeaderLabels(["EXPLORER"])
        self.mw_ui.treeWidget.setIconSize(QSize(25, 25))
        self.mw_ui.treeWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mw_ui.treeWidget.itemClicked.connect(self.__get_selected_file)
        dirs = os.listdir(path)

        fileInfo = QFileInfo(path)
        fileIcon = QFileIconProvider()
        icon = QtGui.QIcon(fileIcon.icon(fileInfo))
        root = QTreeWidgetItem(self.mw_ui.treeWidget)
        root.setText(0, path.split('/')[-1])
        root.setIcon(0, QtGui.QIcon(icon))

        self.__create_tree(dirs, root, path)
        QApplication.processEvents()

    def __get_selected_file(self):
        # 获取当前treewidget选中的文件路径
        item = self.mw_ui.treeWidget.currentItem()
        if item and item.text(0).lower().endswith(('.png', '.jpg', '.bmp')):
            path = item.text(0)
            parent = item.parent()
            while parent is not None:
                path = parent.text(0) + '/' + path
                parent = parent.parent()
            self.cls_input_image_path = f"datasets/" + path #9.29改动 去掉了f{数据集名称}
            # 在预览窗口中显示选中的图片
            add_new_pic_to_display_widget(QPixmap(self.cls_input_image_path), self.mw_ui.preview_pic_widget)

    def __create_tree(self, dirs, root, path):
        for i in dirs:
            path_new = path + '\\' + i
            if os.path.isdir(path_new):
                fileInfo = QFileInfo(path_new)
                fileIcon = QFileIconProvider()
                icon = QtGui.QIcon(fileIcon.icon(fileInfo))
                child = QTreeWidgetItem(root)
                child.setText(0, i)
                child.setIcon(0, QtGui.QIcon(icon))
                dirs_new = os.listdir(path_new)
                self.__create_tree(dirs_new, child, path_new)
            else:
                fileInfo = QFileInfo(path_new)
                fileIcon = QFileIconProvider()
                icon = QtGui.QIcon(fileIcon.icon(fileInfo))
                child = QTreeWidgetItem(root)
                child.setText(0, i)
                child.setIcon(0, QtGui.QIcon(icon))

    def __calibrate(self):#calibrate进train集
        idx = self.mw_ui.comboBox_calibration.currentText()
        self.__confirm_classify(idx)
        
    def __calibrate_val(self):#calibrate进val集
        idx = self.mw_ui.comboBox_calibration.currentText()
        self.__confirm_classify_val(idx)
        
    def __calibrate_test(self):#calibrate进test集
        idx = self.mw_ui.comboBox_calibration.currentText()
        self.__confirm_classify_test(idx)

    def __delpic(self):
        if self.cls_input_image_path is None:
            QMessageBox.critical(self.mw_widget, "Error", "Please select a picture first")
            return
        reply = QMessageBox.information(self.mw_widget, "Delete", "Confirm to delete the selected picture?"
                                        f"\n{self.cls_input_image_path}",
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            os.remove(self.cls_input_image_path)
            self.cls_input_image_path = None
            self.mw_ui.preview_pic_widget.layout().removeItem(self.mw_ui.input_pic_widget.layout().itemAt(0))
            self.__filetree_init()

    def __confirm_classify(self, idx):#将图片归类进train集的对应分类中
        self.__get_model_config()
        if self.cls_input_image_path is None:
            QMessageBox.critical(self.mw_widget, "Error", "Please select a picture first")
            return

        if not os.path.exists(f"datasets/{self.model_config['dataset']}/train/{idx}"):
            os.makedirs(f"datasets/{self.model_config['dataset']}/train/{idx}")

        shutil.copy2(self.cls_input_image_path, f"datasets/{self.model_config['dataset']}/train/{idx}")
        QMessageBox.information(self.mw_widget, "Success", f"Picture moved to train/{idx} successfully")
        self.__filetree_init()
    
    def __confirm_classify_val(self, idx):#将图片归类进val集的对应分类中
        self.__get_model_config()
        if self.cls_input_image_path is None:
            QMessageBox.critical(self.mw_widget, "Error", "Please select a picture first")
            return
        if not os.path.exists(f"datasets/{self.model_config['dataset']}/val/{idx}"):
            os.makedirs(f"datasets/{self.model_config['dataset']}/val/{idx}")
        shutil.copy2(self.cls_input_image_path, f"datasets/{self.model_config['dataset']}/val/{idx}")
        QMessageBox.information(self.mw_widget, "Success", f"Picture moved to val/{idx} successfully")
        self.__filetree_init()
        
    def __confirm_classify_test(self, idx):#将图片归类进test集的对应分类中
        self.__get_model_config()
        if self.cls_input_image_path is None:
            QMessageBox.critical(self.mw_widget, "Error", "Please select a picture first")
            return
        if not os.path.exists(f"datasets/{self.model_config['dataset']}/test/{idx}"):
            os.makedirs(f"datasets/{self.model_config['dataset']}/test/{idx}")
        shutil.copy2(self.cls_input_image_path, f"datasets/{self.model_config['dataset']}/test/{idx}")
        QMessageBox.information(self.mw_widget, "Success", f"Picture moved to test/{idx} successfully")
        self.__filetree_init()        

    def __update_acc(self):
        self.cls_result = self.cls_thread.get_res()
        self.mw_ui.label_nametop1.setText(self.cls_result[0].names[self.cls_result[0].probs.top5[0]])
        
        self.mw_ui.label_nametop2.setText(self.cls_result[0].names[self.cls_result[0].probs.top5[1]])
        self.mw_ui.label_nametop3.setText(self.cls_result[0].names[self.cls_result[0].probs.top5[2]])
        self.mw_ui.label_nametop4.setText(self.cls_result[0].names[self.cls_result[0].probs.top5[3]])
        self.mw_ui.label_nametop5.setText(self.cls_result[0].names[self.cls_result[0].probs.top5[4]])
        
        self.mw_ui.label_numtop1.setText(f"{self.cls_result[0].probs.top5conf[0]:.2%}")#保留2位小数并以%方式输出
        
        self.mw_ui.label_numtop2.setText(f"{self.cls_result[0].probs.top5conf[1]:.2%}")
        self.mw_ui.label_numtop3.setText(f"{self.cls_result[0].probs.top5conf[2]:.2%}")
        self.mw_ui.label_numtop4.setText(f"{self.cls_result[0].probs.top5conf[3]:.2%}")
        self.mw_ui.label_numtop5.setText(f"{self.cls_result[0].probs.top5conf[4]:.2%}")
        
    def __begin_classify(self):
        if self.cls_input_image_path is None:
            QMessageBox.critical(self.mw_widget, "Error", "Please select a picture first")
            return
        if self.mw_ui.comboBox_model.currentIndex() == 0:
            QMessageBox.critical(self.mw_widget, "Error", "Please select a model first")
            return
        model_path = os.path.join('runs/classify', self.mw_ui.comboBox_model.currentText(), 'weights/best.pt')
        self.cls_thread = ClassifyThread(model_path, self.cls_input_image_path)#调用detect.py的方法进行模型预测
        self.cls_thread.finished.connect(self.__update_acc)
        self.cls_thread.start()

    def __get_picture_from_file_dialog(self, picwidget : QWidget = None):
        file_name, _ = QFileDialog.getOpenFileName(self.mw_widget, "Open Image", self.lastfile_path,
                                                   "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.cls_input_image_path = file_name
            self.last_file_dialog_path = os.path.dirname(file_name)
            add_new_pic_to_display_widget(QPixmap(self.cls_input_image_path), picwidget)

    def __detect_reset(self):

        self.mw_ui.label_numtop1.setText("")
        
        self.mw_ui.label_numtop2.setText("")
        self.mw_ui.label_numtop3.setText("")
        self.mw_ui.label_numtop4.setText("")
        self.mw_ui.label_numtop5.setText("")
        
        self.mw_ui.label_nametop1.setText("")
        
        self.mw_ui.label_nametop2.setText("")
        self.mw_ui.label_nametop3.setText("")
        self.mw_ui.label_nametop4.setText("")
        self.mw_ui.label_nametop5.setText("")
        
    def __update_comboBox_model(self):
        self.exist_model_save_name = get_all_file_name('runs/classify')
        self.mw_ui.comboBox_model.clear()
        self.mw_ui.comboBox_model.addItems([name for name in ["None"] + self.exist_model_save_name])
        self.__detect_reset()

    def __visualizer_init(self, model_name: str = ''):#tensorboard调用
        self.log_path += f"/{model_name}"   
        if self.tb is None:
            self.tb = program.TensorBoard()
            
        self.tb.configure(argv=[None, '--logdir', f'{self.log_path}'])
        self.tb.launch()#启动tensorboard
        self.mw_ui.vis_wid.layout().removeWidget(self.web_view)#清空网页
        self.web_view = QWebEngineView()
        self.web_view.load(QUrl("http://localhost:6006/#scalars"))
        self.mw_ui.vis_wid.layout().addWidget(self.web_view)#添加网页
        self.mw_ui.pushButton_refreshBrowser.clicked.connect(lambda: (
            self.web_view.reload(),
            print("Reload browser")
        ))
    #在此获取configuration
    def __get_model_config(self):#获取模型参数
        self.model_config['weight'] ="yolov8x-cls" #self.mw_ui.comboBox_weights.currentText() 基础模型依据文本框内容
        self.model_config['epochs'] = self.mw_ui.spinBox_epochs.value()
        self.model_config['batch_size'] = self.mw_ui.spinBox_batchSize.value()
        self.model_config['lr'] = self.mw_ui.doubleSpinBox.value()
        self.model_config['optimizer'] = self.mw_ui.comboBox_optimizer.currentText()
        self.model_config['device'] = 'cuda' if self.mw_ui.comboBox_device.currentText() == 'GPU' else 'cpu'
        self.model_config['imgsz'] = self.mw_ui.spinBox_imgSize.value()
        self.model_config['workers'] = self.mw_ui.spinBox_workers.value()
        self.model_config['dataset'] = self.mw_ui.comboBox_dataset.currentText()

    def __on_train_finished(self):
        QMessageBox.information(self.mw_widget, "Train", "Training finished")
        self.__update_comboBox_model()
        print("Refresh browser")

    def __train_model(self):
        self.__get_model_config()
        print(self.model_config)
        reply = QMessageBox.information(self.mw_widget, "Train", "Confirm to start training at this configuration?"
                                                                 "Classify widget will be reset.",
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return

        self.model_name = f"Weight_{self.model_config['weight']}_Epochs_{self.model_config['epochs']}_"\
                         f"Batch_{self.model_config['batch_size']}_Lr_{self.model_config['lr']}_"\
                         f"Loss_{self.model_config['optimizer']}_Opt_{self.model_config['device']}_"\
                         f"Imgsz_{self.model_config['imgsz']}_Workers_{self.model_config['workers']}_"\
                         f"Dataset_{self.model_config['dataset']}_"

        self.train_thread = TrainThread(self.model_config, self.model_name)

        self.train_thread.finished.connect(self.__on_train_finished)
        self.train_thread.start()

    def show(self):
        self.mw_widget.show()

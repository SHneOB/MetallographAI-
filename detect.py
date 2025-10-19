from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal, QObject


class ClassifyThread(QThread):

    def __init__(self, model_path: str, cls_input_image_path: str, parent=None):
        super().__init__(parent)
        #模型路径
        self.model_path = model_path
        #预测图片路径
        self.cls_input_image_path = cls_input_image_path
        #结果初始化为None
        self.result = None

    def run(self):
        model = YOLO(f"{self.model_path}")#加载YOLO模型
        #进行预测
        self.result = model.predict(source=self.cls_input_image_path,project='runs/detect',
                                    show_labels=True, show_conf=True, save_txt=True, save_conf=True, save_crop=True)
        #输出预测结果
        print(self.result)

    def get_res(self):
        return self.result

from contextlib import redirect_stdout

from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from io import StringIO


class TrainThread(QThread):

    def __init__(self, model_config: dict, model_name: str, parent=None):
        super().__init__(parent)
        self.model_config = model_config
        self.model_name = model_name

        self.result = None

    def run(self):
        model = YOLO(f"{self.model_config['weight_path']}/{self.model_config['weight']}.pt")

        self.result = model.train(data=self.model_config['dataset'], epochs=self.model_config['epochs'],
                                  batch=self.model_config['batch_size'], lr0=self.model_config['lr'],
                                  device=self.model_config['device'], optimizer=self.model_config['optimizer'],
                                  name=self.model_name, imgsz=self.model_config['imgsz'],
                                  workers=self.model_config['workers'],)

    def get_res(self):
        return self.result

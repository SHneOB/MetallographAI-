from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel

import os
import re


def get_all_file_name(file_path: str) -> list[str]:
    return os.listdir(f"./{file_path}")


def get_info_from_saved_model(filename: str) -> dict:
    pattern = r"Weight(.*)_Epochs_(.*)_Batch_(.*)_Lr_(.*)_Loss_(.*)_Opt_(.*)" \
              r"_Device_(.*)_Author_(.*)"
    match = re.match(pattern, filename)
    if match:
        return {
            'model': match.group(1),
            'epochs': match.group(2),
            'batch_size': match.group(3),
            'lr': match.group(4),
            'loss': match.group(5),
            'optimizer': match.group(6),
            'device': match.group(8),
            'user': match.group(9),
        }
    else:
        return {
            'model': 'None',
            'epochs': 'None',
            'batch_size': 'None',
            'lr': 'None',
            'loss': 'None',
            'optimizer': 'None',
            'device': 'None',
            'user': 'None',
        }


def add_new_pic_to_display_widget(picture: QPixmap, widget: QWidget):
    widget.layout().removeItem(widget.layout().itemAt(0))
    pic_label = QLabel()
    pic_label.setPixmap(picture)
    pic_label.setScaledContents(True)
    pic_label.setAlignment(Qt.AlignCenter)
    widget.layout().addWidget(pic_label)

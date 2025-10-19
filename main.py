from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from system import System
import sys

import qdarkstyle

# 读取自定义的 .qss 文件  
def load_custom_stylesheet(qss_filepath):  
    with open(qss_filepath, 'r', encoding='utf-8') as file:  
        return file.read()  
#如果当前为主程序
if __name__ == '__main__':

    # 设置高DPI缩放
    #QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    
    # 加载 qdarkstyle 样式表  
    qdark_stylesheet = qdarkstyle.load_stylesheet_pyqt5()  
  
    # 加载自定义的 .qss 文件  
    custom_stylesheet = load_custom_stylesheet('./style.qss')  
  
    # 合并样式表  
    merged_stylesheet = qdark_stylesheet + custom_stylesheet  
  
    # 设置合并后的样式表给 QApplication  
    app.setStyleSheet(merged_stylesheet)  
    
    system = System()
    system.show()

    sys.exit(app.exec_())
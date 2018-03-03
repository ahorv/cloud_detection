import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class QCustomLabel (QLabel):
    def __init__ (self, parent = None):
        super(QCustomLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.setTextLabelPosition(0, 0)
        self.setAlignment(Qt.AlignCenter)

    def mouseMoveEvent (self, eventQMouseEvent):
        self.setTextLabelPosition(eventQMouseEvent.x(), eventQMouseEvent.y())
        QWidget.mouseMoveEvent(self, eventQMouseEvent)

    def mousePressEvent (self, eventQMouseEvent):
        if eventQMouseEvent.button() == Qt.LeftButton:
            QMessageBox.information(self, 'Position', '( %d : %d )' % (self.x, self.y))
            QWidget.mousePressEvent(self, eventQMouseEvent)

    def setTextLabelPosition (self, x, y):
        self.x, self.y = x, y
        self.setText('Please click on screen ( %d : %d )' % (self.x, self.y))

class QCustomWidget (QWidget):
    def __init__ (self, parent = None):
        super(QCustomWidget, self).__init__(parent)
        self.setWindowOpacity(0.7)
        # Init QLabel
        self.positionQLabel = QCustomLabel(self)
        # Init QLayout
        layoutQHBoxLayout = QHBoxLayout()
        layoutQHBoxLayout.addWidget(self.positionQLabel)
        #layoutQHBoxLayout.setMargin(0)
        #layoutQHBoxLayout.setSpacing(0)
        self.setLayout(layoutQHBoxLayout)
        self.showFullScreen()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myQTestWidget = QCustomWidget()
    myQTestWidget.show()
    sys.exit(app.exec_())

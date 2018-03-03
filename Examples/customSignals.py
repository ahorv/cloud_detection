import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class MyWidget(QWidget):

    clicked = pyqtSignal()
    keyPressed = pyqtSignal()

    def __init__(self, parent = None):

       QWidget.__init__(self, parent)
       self.color = QColor(0, 0, 0)

    def paintEvent(self, event):

       painter = QPainter()
       painter.begin(self)
       painter.fillRect(event.rect(), QBrush(self.color))
       painter.end()

    def keyPressEvent(self, event):

       self.keyPressed.emit(event.text())
       event.accept()

    def mousePressEvent(self, event):

       self.setFocus(Qt.OtherFocusReason)
       event.accept()

    def mouseReleaseEvent(self, event):

       if event.button() == Qt.LeftButton:

           self.color = QColor(self.color.green(), self.color.blue(),
                               127 - self.color.red())
           self.update()
           self.clicked.emit()
           event.accept()

    def sizeHint(self):

       return QSize(100, 100)


if __name__ == "__main__":

   app = QApplication(sys.argv)
   window = QWidget()

   mywidget = MyWidget()
   label = QLabel()

   mywidget.clicked.connect(label.clear)
   mywidget.keyPressed.connect(label.setText)

   layout = QVBoxLayout()
   layout.addWidget(mywidget)
   layout.addWidget(label)
   window.setLayout(layout)

   window.show()
   sys.exit(app.exec_())


from helper import *

from PyQt5.QtCore import QTimer, QPoint, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor

import queue as Queue

class Window(QMainWindow):
    text_update = pyqtSignal(str)

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        QMainWindow.setFixedSize(self, 660, 600)
        self.central = QWidget(self)
        self.textbox = QTextEdit(self.central)
        self.textbox.setFont(FONT)
        self.textbox.setMaximumSize(640, 100)
        self.textbox.setReadOnly(True)
        self.textbox.setAlignment(Qt.AlignLeft)
        self.text_update.connect(self.append_text)
        sys.stdout = self
        print("Output:")
        
        self.vlayout = QVBoxLayout()    
        self.displays = QHBoxLayout()
        self.disp = ImageWidget(self)    
        self.displays.addWidget(self.disp)
        self.vlayout.addLayout(self.displays)
        self.vlayout.addWidget(self.textbox)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
 
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)
        self.addAction(exitAction)

    def start(self):
        self.timer = QTimer(self)         
        self.timer.timeout.connect(lambda: 
                    self.show_image(img_queue, self.disp))
        self.timer.start(DISP_MSEC)         
        self.capture_thread = threading.Thread(target=main, 
                    args=(CAM_NUM, img_queue))
        self.capture_thread.start()         

    def show_image(self, imageq, display):
        if not imageq.empty():
            image = imageq.get()
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display)

    def display_image(self, img, display):
        qimg = QImage(img.data,img.shape[1],img.shape[0],FORMAT)
        display.setImage(qimg)

    def write(self, text):
        self.text_update.emit(str(text))
        
    def flush(self):
        pass

    def append_text(self, text):
        cur = self.textbox.textCursor()    
        cur.movePosition(QTextCursor.End) 
        s = str(text)
        while s:
            head,sep,s = s.partition("\n") 
            cur.insertText(head)           
            if sep:                      
                cur.insertBlock()
        self.textbox.setTextCursor(cur)    

    def closeEvent(self, event):
        global capture
        capture = False
        self.capture_thread.join()

class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()

def main(cam_num, queue):
   
    frame_rate_calc = 1
    time1  = 0
    cap = WebcamVideoStream(src=CAM_NUM, res=RES, fps=FPS, api= API).start()
#     cap = cv2.VideoCapture(CAM_NUM, cv2.CAP_V4L2)
    while capture:
  
        try:
            start = time.perf_counter()
            _,image = cap.read()
#             image = cv2.flip(image, 1)
                        
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (w, h))
            input_data = np.expand_dims(frame_resized, axis=0).astype('uint8')
            
            results = detect_objects(interpreter, input_data, threshold)
            image = annotate_objects(image, results, RES[0], RES[1], labels, time1 ,frame_rate_calc)
            
            time1 = (time.perf_counter()-start)
            frame_rate_calc = 1/time1
               
            image =cv2.resize(image, IMG_SIZE)
            if image is not None and queue.qsize() < 5:
                queue.put(image)
                
            else:
                time.sleep(DISP_MSEC / 1000.0)
        
        except Exception as e:
            print("Failed")
            print(e)
    
    cap.stop()
    del cap
    time.sleep(2)

if __name__ == '__main__':
    
    img_queue    = Queue.Queue()
    FORMAT       = QImage.Format_RGB888
    DISP_MSEC    = 10               
    FONT         = QFont("Helvetica", 10)
    capture      = True
    
    IMG_SIZE     = (640, 480)
    RES          = (640, 480)
    
    CAM_NUM      = 0
    API          = cv2.CAP_V4L2
    FPS          = 30
   
    USE_TPU      = 0
    NUM_THREAD   = 4
    threshold    = 0.5

    GRAPH_FOLDER = 'export/SSD/5/150l/tflite/'
    GRAPH_NAME   = 'detect_model.tflite'
    LABEL_NAME   = 'data/TFLite_label.txt'

    result = searchFile(GRAPH_FOLDER)
    interpreter = make_interpreter(result, NUM_THREAD)
    time.sleep(2)

    _, h, w, _ = interpreter.get_input_details()[0]['shape']

    labels = load_tflite_label(LABEL_NAME)
    print(labels)
    
    TITLE = "Traffic Sign Detection Using Raspberry Pi 4"
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    win.setWindowTitle(TITLE)
    win.start()
    sys.exit(app.exec_())
    sys.exit(0)



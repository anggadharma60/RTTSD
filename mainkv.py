

from kivy.app import App


from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from kivy.clock import mainthread
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.config import Config
Config.set('graphics','resizable', False)
Config.set('graphics', 'width', '660')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'position', 'auto')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
# Config.write()


from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window

import cv2
import time
import os
from helper import *

Builder.load_string('''
<Main>:
    id:main
    BoxLayout:  
        
        orientation: 'vertical'
        size: 660, 600
        padding: 10
        spacing: 10
        canvas.before:
            Color:
                rgba: (1,1,1,0.75)
            Rectangle:
                size: self.size
                pos: self.pos
        
        Image:
            id: image
            size: 640,480
            # size: self.texture_size
            # source: "00000.png"

        
        Label:
            id: label
            text: "Output"
            size_hint_y:.2
            font_size: 20
            color: 0,0,0,1

            background_color: (1,1,1,1)
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
    
        # Button:
        #     id:capture
        #     text: 'Capture'
        #     size_hint_y:.1
        #     font_size: 20
        #     border: (16,16,16,16)
        #     on_press: root.screenshot()
        #     background_color: (0.5,0.5,0.5,1)
        

''')

class AppCam(Image):
    IMG  = (640, 480)
    threshold    = 0.5
    frame_rate_calc = 1
    time1  = 0
    
    def __init__(self, parent, capture, **kwargs):
        super(AppCam, self).__init__(**kwargs)
        self.parent = parent
        self.capture = capture
        Logger.info('Application: Succesfully load camera')
        
    def start(self, fps=30):
        Clock.schedule_interval(self.update, 1.0 / self.parent.FPS)
    
    
    def update(self, dt):
        
        _, h, w, _ = self.parent.interpreter.get_input_details()[0]['shape']
        ret, self.frame = self.capture.read()
     
        if ret:
            start = time.perf_counter()
            
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (w, h))
            input_data = np.expand_dims(frame_resized, axis=0).astype('uint8')
            results = self.detect_objects(self.parent.interpreter, input_data, self.threshold)

            self.annotateObject(self.frame, results, self.parent.RES[0],self.parent.RES[1], self.parent.labels, self.time1 ,self.frame_rate_calc)
            
            self.time1 = (time.perf_counter()-start)
            self.frame_rate_calc = 1/self.time1
            
            image = cv2.resize(self.frame, (self.IMG[0],self.IMG[1]))
            buf = cv2.flip(image, 0).tobytes()
            
            image_texture = Texture.create(size=(self.IMG[0],self.IMG[1]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
            self.parent.ids['image'].texture = self.texture
    
         
    def detect_objects(self, interpreter, image, threshold):
        set_input_tensor(interpreter, image)
        interpreter.invoke()
          
        boxes = get_output_tensor(interpreter, 0)
        classes = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)
        count = int(get_output_tensor(interpreter, 3))
          
        results = []
        for i in range(count):
            if scores[i] >= threshold:
          
              result = {
                  'bounding_box': boxes[i],
                  'class_id': classes[i],
                  'score': scores[i]
              }
              results.append(result)
          
        return results
    
    
    def annotateObject(self, frame, results, WIDTH, HEIGHT, labels, time1, frame_rate_calc):
        color_box = [(0,0,255), (255,0,0), (0,255,255), (255,255,0)]
        hide_led()
        for obj in results:
            
            ymin, xmin, ymax, xmax = obj['bounding_box']

            ymin = int(max(1, (ymin * HEIGHT)))
            xmin = int(max(1, (xmin * WIDTH)))
            ymax = int(min(HEIGHT, (ymax * HEIGHT)))
            xmax = int(min(WIDTH, (xmax * WIDTH)))

            cl = int(obj['class_id']+1)
            classid =labels[cl-1]
            score = round(obj['score'],2)

            arr = ('%4d %4d %4d %4d %d %-11s %.2f %.3fs %.2f' % (xmin, ymin,
            xmax, ymax, cl, classid, score, time1, frame_rate_calc))
            

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color_box[cl-1], 2)
            cv2.putText(frame, '%d: %s %.2f' % (cl, classid, score ), (
              xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box[cl-1], 2)

            self.parent.updateLabel(arr)
            show_led(cl)

    
    def stop(self):
        Clock.unschedule(self.update)
        
        
class Main(BoxLayout):
    
    ssFolder = os.path.join(os.getcwd(),'screenshot')
    appCam = None
    capture = None
    interpreter  = None
    labels = None
    
    def __init__(self, CAM_NUM, RES, FPS, API, **kwargs):
        super(Main, self).__init__(**kwargs)
        self.CAM_NUM = CAM_NUM
        self.RES = RES
        self.FPS = FPS
        self.API = API
    
    
    def running(self, interpreter, labels):
        self.interpreter = interpreter
        self.labels = labels
#         
#         self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
#         self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.RES[0])
#         self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.RES[1])
#         self.capture.set(cv2.CAP_PROP_FPS, self.FPS)
# 
        self.capture = WebcamVideoStream(src=self.CAM_NUM, res=self.RES, fps=self.FPS, api= self.API).start()
        self.appCam = AppCam(self, self.capture)
        self.appCam.start()
          
    def updateLabel(self, arr):
        self.ids.label.text = arr
        
    def stopped(self): 
#         self.capture.release()
        
        self.capture.stop()
#         self.capture.stream.release()
      
      
class Apps(App):
    
    GRAPH_FOLDER = 'export/SSD/5/150l/tflite/'
    GRAPH_NAME   = 'detect_model.tflite'
    LABEL_NAME   = 'data/TFLite_label.txt'
    
    CAM_NUM      = 0
    API          = cv2.CAP_V4L2
    USE_TPU      = 0
    FPS          = 30
    RES          = (640, 480)
    NUM_THREAD   = 4
    

    mainLayout   = None
    interpreter  = None
    labels       = None     
    
    
    def build(self):
        #Set Title and Icon
        self.title = "Traffic Sign Detection Using Raspberry Pi 4"
        self.icon = "Undip.png"
        
        #Load interpreter
        result = searchFile(self.GRAPH_FOLDER)
        self.interpreter = make_interpreter(result, self.NUM_THREAD)
        Logger.info('Application: Succesfully load Intepreter '+str(self.interpreter))
        
        #load labels
        self.labels = load_tflite_label(self.LABEL_NAME)
        Logger.info('Application: Succesfully load label '+str(self.labels))
        
        time.sleep(1)

        #load Interface
        self.mainLayout = Main(self.CAM_NUM, self.RES, self.FPS, self.API)
        # self.mainLayout = Main()
        self.mainLayout.running(self.interpreter, self.labels)
        return self.mainLayout
    
    def on_stop(self):
        
        self.mainLayout.stopped()
        poweroff_led()

        
    
if __name__ == "__main__":
    
    apps = Apps()
    threading.Thread(target=apps.run())
    sys.exit(0)
    

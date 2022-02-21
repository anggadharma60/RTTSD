import os
import cv2
import numpy as np
import re
import sys
import time
import importlib.util
from PIL import Image, ImageDraw, ImageFont
import platform
from threading import Thread
from led import *
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

class WebcamVideoStream:
    def __init__(self, src=0, res=(640,480), fps=30, api=cv2.CAP_V4L2,name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src, api)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
  
        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return True,self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
      
    set_input_tensor(interpreter, image)
    interpreter.invoke()
      
    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
       
    results = []
    max_object = min(count, 10)
    for i in range(max_object):
        if scores[i] >= threshold:
      
          result = {
              'bounding_box': boxes[i],
              'class_id': classes[i],
              'score': scores[i]
          }
          results.append(result)
      
    return results

def annotate_objects(frame, results, WIDTH, HEIGHT, labels, time1, frame_rate_calc):
    color_box = [(0,0,255), (255,0,0), (0,255,255), (255,255,0)]
    hide_led()
    for obj in results:
    
      ymin, xmin, ymax, xmax = obj['bounding_box']
    
      ymin = int(max(1, (ymin * HEIGHT)))
      xmin = int(max(1, (xmin * WIDTH)))
      ymax = int(min(HEIGHT, (ymax * HEIGHT)))
      xmax = int(min(WIDTH, (xmax * WIDTH)))
      
      cl = int(obj['class_id'])
      classid =labels[cl]
      score = obj['score']
      
      print('%d %d %d %d %d %s %.2f %fs %.1f' % (xmin, ymin, xmax, ymax, cl, classid, score, time1, frame_rate_calc))
      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color_box[cl], 2)
      cv2.putText(frame, '%d: %s %.2f' % (cl, classid, score ), (
          xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box[cl], 2)
      show_led(cl)
    
    return frame


def check_platform():
    return platform.system()


def searchFile(folder):
    result = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".tflite"):
                result.append(os.path.join(os.getcwd(), root, file))

    return result

def make_interpreter(result, NUM_THREAD):
    system = check_platform()
    edgetpu=None
    try:
        if platform == "Windows":
            edgetpu = load_delegate('edgetpu.dll')
        if platform == "Linux":
            edgetpu = load_delegate('libedgetpu.so.1')
    except:
        pass
    if edgetpu is None:
        print("Edge TPU Not Detected")
        interpreter = Interpreter(model_path=result[0], num_threads=NUM_THREAD)
    else:
        print("Edge TPU Detected")
        interpreter = Interpreter(model_path=result[1], num_threads=NUM_THREAD,
                                      experimental_delegates=[edgetpu])
    interpreter.allocate_tensors()
    return interpreter


def load_tflite_label(path):
    folder_path = os.path.join(os.getcwd(), path)
    with open(folder_path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      ret = {}
      for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
    
        if len(pair) == 2 and pair[0].strip().isdigit():
          ret[int(pair[0])] = pair[1].strip()
    
        else:
          ret[row_number] = pair[0].strip()
          # print(pair[0].strip())
    return ret


def main(CAM_NUM, WIDTH, HEIGHT, GRAPH_FOLDER, LABEL_NAME, threshold, API):

    labels = load_tflite_label(LABEL_NAME)
    result = searchFile(GRAPH_FOLDER)

    interpreter = make_interpreter(result, NUM_THREAD)

    _, h, w, _ = interpreter.get_input_details()[0]['shape']

    frame_rate_calc = 1
    time1 = 0
    freq = cv2.getTickFrequency()
    videostream = WebcamVideoStream(src=DEVICE, res=(
        WIDTH, HEIGHT), fps=30, api = API).start()
#     videostream = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

    

    while True:
        try:
            t1 = cv2.getTickCount()

            _,image = videostream.read()
#             frame = cv2.flip(frame, 1)
            frame =image.copy()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (w, h))
            input_data = np.expand_dims(frame_resized, axis=0)
            input_data

            results = detect_objects(interpreter, input_data, threshold)
            
            res = annotate_objects(frame, results, WIDTH, HEIGHT, labels, time1, frame_rate_calc)

            cv2.putText(res, 'FPS: %.1f' % (frame_rate_calc),
                        (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            
            cv2.imshow("", res)

            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:

            break

    cv2.destroyAllWindows()
    videostream.stop()
    time.sleep(2)


if __name__ == '__main__':

    GRAPH_FOLDER = 'export/SSD/5/150l/tflite/'
    GRAPH_NAME = 'detect_model.tflite'
    LABEL_NAME = 'data/TFLite_label.txt'
    USE_TPU = 0
    DEVICE = 0
    WIDTH = 640
    HEIGHT = 480
    threshold = float(0.5)
    NUM_THREAD = 4
    API = cv2.CAP_V4L2
    main(DEVICE, WIDTH, HEIGHT, GRAPH_FOLDER,
         LABEL_NAME, threshold, API)
    sys.exit()

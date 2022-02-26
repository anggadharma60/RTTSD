import numpy as np
import sys, time, threading, cv2, os, re, platform, importlib
from threading import Thread

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from led import *

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
    platform = check_platform()
    edgetpu= None
    try:
        if platform == "Windows":
            edgetpu = load_delegate('edgetpu.dll')
        if platform == "Linux":
            edgetpu = load_delegate('libedgetpu.so.1')
            print("halo")
    except:
        pass
    if edgetpu is None:
        print("Edge TPU Not Detected")
        NUM_THREAD = 2
        interpreter = Interpreter(model_path=result[0], num_threads=NUM_THREAD)
    else:
        print("Edge TPU Detected")
        interpreter = Interpreter(model_path=result[1], num_threads=NUM_THREAD,
                                      experimental_delegates=[edgetpu])
    interpreter.allocate_tensors()
    return interpreter
        
def build_interpreter(result, USE_TPU, NUM_THREAD):
    platform = check_platform()
    pkg = importlib.util.find_spec('tflite_runtime')
    # pkg = importlib.util.find_spec('tensorflow')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if USE_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if USE_TPU:
            from tensorflow.lite.python.interpreter import load_delegate
    if USE_TPU:
        if (platform == "Linux"):
            interpreter = Interpreter(model_path=result[1], num_threads=NUM_THREAD,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1')])
    
        if (platform == "Windows"):
            interpreter = Interpreter(model_path=result[1], num_threads=NUM_THREAD,
                                      experimental_delegates=[load_delegate('edgetpu.dll')])
    else:
        interpreter = Interpreter(model_path=result[0], num_threads=NUM_THREAD)
    
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
    
    if isinstance(results, type(None)):
        return frame
    
    for obj in results:
    
      ymin, xmin, ymax, xmax = obj['bounding_box']
      
      ymin = int(max(1, (ymin * HEIGHT)))
      xmin = int(max(1, (xmin * WIDTH)))
      ymax = int(min(HEIGHT, (ymax * HEIGHT)))
      xmax = int(min(WIDTH, (xmax * WIDTH)))
      
      cl = int(obj['class_id']+1)
      classid =labels[cl-1]
      score = round(obj['score'],2)
        
      print('%4d %4d %4d %4d %d %-11s %.2f %.3fs %.2f' % (xmin, ymin,
      xmax, ymax, cl, classid, score, time1, frame_rate_calc))

      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color_box[cl-1], 2)
      cv2.putText(frame, '%d: %s %.2f' % (cl, classid, score ), (
          xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box[cl-1], 2)
      
      show_led(cl)
    
    return frame

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

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
        self.stream.release()
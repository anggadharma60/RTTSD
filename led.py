from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text

import time

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, rotate=2)

def show_led(cl):
    classid = str(cl)
    with canvas(device) as draw:
        text(draw, (1, 0), classid, fill="white")
    time.sleep(1/1000)
    

def hide_led():
    with canvas(device) as draw:
        text(draw, (0, 0), "", fill="white")
    time.sleep(1/1000)
    
def poweroff_led():
    device.cleanup()
    


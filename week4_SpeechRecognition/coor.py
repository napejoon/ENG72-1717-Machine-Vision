from pydobot import Dobot
import time

device = Dobot(port='COM10')
print("เชื่อมต่อแล้ว")

#X=200 Y=0 Z=50 R=0
device.move_to(150, 0, 50, 0, wait=True)
print("เคลื่อนที่")

device.suck(True)
print("ดูด")
time.sleep(1)


device.suck(False)
print("หยุดดูด")
time.sleep(1)
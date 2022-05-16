#!/usr/bin/python3
import time
import board
import adafruit_fxas21002c

i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_fxas21002c.FXAS21002C(i2c)

while True:
    gyro_x, gyro_y, gyro_z = sensor.gyroscope
    print('Gyroscope (radians/s): ({0:0.3f},  {1:0.3f},  {2:0.3f})'.format(gyro_x, gyro_y, gyro_z))
    time.sleep(1.0)

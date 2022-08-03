#!/usr/bin/python3
import time
import board
import adafruit_fxos8700

i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_fxos8700.FXOS8700(i2c)

while True:
    accel_x, accel_y, accel_z = sensor.accelerometer
    mag_x, mag_y, mag_z = sensor.magnetometer
    #print('Acceleration (m/s^2): ({0:0.3f}, {1:0.3f}, {2:0.3f})'.format(accel_x, accel_y, accel_z))
    print('Magnetometer (uTesla): ({0:0.3f}, {1:0.3f}, {2:0.3f})'.format(mag_x, mag_y, mag_z))
    time.sleep(1.0)

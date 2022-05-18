#!/usr/bin/python3

# Heliostat driver - this script should be run as a service

# In the end, this script should provide the remote IoT interface

# During development, it's invoked on command line

import adafruit_fxos8700
import board
from consolemenu import *
from consolemenu.items import *
import csv
from gpiozero import Motor
import logging
from scipy.signal import savgol_filter
import time

# Setup logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d %(funcName)s] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)

# Setup motors
MOTOR_TRAVEL_TIME = 30  # seconds
MOTORH = Motor(26, 20)
MOTORA = Motor(19, 16)

# Setup sensor
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_fxos8700.FXOS8700(i2c)

# Setup calibration data global variable
cal = { 'acxs': [], 'acys': [], 'aczs': [], 'tcxss': [], 'tcyss': [], 'tczss': [] }


def calibrate():
    global cal
    logging.info("Start calibration")
    logging.info("Start fully retract azimut motor")
    MOTORA.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract azimut motor")
    MOTORA.stop()
    cal = { 'acxs': [], 'acys': [], 'aczs': [], 'tcxss': [], 'tcyss': [], 'tczss': [] }
    remaining_amotor_travel_time = MOTOR_TRAVEL_TIME
    partial_amotor_travel_time = MOTOR_TRAVEL_TIME / 1
    while remaining_amotor_travel_time > 0:
        logging.info("Start fully retract heading motor")
        MOTORH.backward()
        time.sleep(MOTOR_TRAVEL_TIME)
        MOTORH.stop()
        logging.info("Stop fully retract heading motor")
        acx, acy, acz = sensor.accelerometer
        tcxs, tcys, tczs = scan()
        cal['acxs'].append(acx)
        cal['acys'].append(acy)
        cal['aczs'].append(acz)
        cal['tcxss'].append(savgol_filter(tcxs, 9, 1))
        cal['tcyss'].append(savgol_filter(tcys, 9, 1))
        cal['tczss'].append(savgol_filter(tczs, 9, 1))
        logging.info("Start partially extend azimut motor")
        MOTORA.forward()
        time.sleep(partial_amotor_travel_time)
        MOTORA.stop()
        remaining_amotor_travel_time -= partial_amotor_travel_time
        logging.info("Stop partially extend azimut motor")
    logging.info("Writing data to file")
    with open('tcxss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(range(len(cal['tcxss'])))
        for tcxs in zip(*cal['tcxss']):
            writer.writerow(tcxs)
    with open('tcyss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(range(len(cal['tcyss'])))
        for tcys in zip(*cal['tcyss']):
            writer.writerow(tcys)
    with open('tczss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(range(len(cal['tczss'])))
        for tczs in zip(*cal['tczss']):
            writer.writerow(tczs)
    with open('acxyzs.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y','z'])
        for axyz in zip(cal['acxs'], cal['acys'], cal['aczs']):
            writer.writerow(axyz)
    logging.info("Start fully retract azimut motor")
    MOTORA.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract azimut motor")
    logging.info("Stop calibration")

def load_calibration():
    global cal



def scan():
    logging.info("Start heading scan")
    tcxs = []
    tcys = []
    tczs = []
    stop = time.time() + MOTOR_TRAVEL_TIME  # seconds
    logging.info("Start full travel extending of heading motor")
    MOTORH.forward()
    while time.time() < stop:
        tcx, tcy, tcz = sensor.magnetometer
        tcxs.append(tcx)
        tcys.append(tcy)
        tczs.append(tcz)
        time.sleep(0.1)
    logging.info("Stop full travel extending of heading motor")
    MOTORH.stop()
    logging.info("Stop heading scan")
    return tcxs, tcys, tczs


def main():
    menu = ConsoleMenu("Heliostat", "Control Center")
    function_item = FunctionItem("Calibrate", calibrate)
    function_item = FunctionItem("Load calibration from file", load_calibration)
    menu.append_item(function_item)
    menu.show()


if __name__ == "__main__":
    main()


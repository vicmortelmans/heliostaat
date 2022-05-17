#!/usr/bin/python3

# Heliostat driver - this script should be run as a service

# In the end, this script should provide the remote IoT interface

# During development, it's invoked on command line

from consolemenu import *
from consolemenu.items import *
from gpiozero import Motor
import time

# Setup
MOTOR_TRAVEL_TIME = 30  # seconds
MOTORH = Motor(26, 20)
MOTORA = Motor(0, 0)


def calibrate():
    MOTORH.forward()
    time.sleep(MOTOR_TRAVEL_TIME)
    MOTORH.stop()
    MOTORA.forward()
    time.sleep(MOTOR_TRAVEL_TIME)
    MOTORA.stop()
    # data =  [[]]
    # acs = []
    # xnrs = []
    while True:
        ac = sensor_gravity_readout()
        tcxs, tcyx, tczs = scan()
        acs.append(ac)
        tcxss.append(tcxs)
        tcyss.append(tcys)
        tczss.append(tczs)
        MOTORH.backward()
        time.sleep(MOTOR_TRAVEL_TIME)
        MOTORH.stop()


def scan():
    tcxs = []
    tcys = []
    tczs = []
    stop = time.time() + MOTOR_TRAVEL_TIME  # seconds
    MOTORH.forward()
    while time.time() < stop:
        tcxs.append(...)
        tcys.append(...)
        tczs.append(...)
    MOTORH.stop()
    return tcxs, tcyx, tczs


def main():
    menu = ConsoleMenu("Heliostat", "Control Center")
    function_item = FunctionItem("Calibrate", calibrate)
    menu.append_item(function_item)
    menu.show()


if __name__ == "__main__":
    main()


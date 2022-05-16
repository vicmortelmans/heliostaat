#!/usr/bin/python3

# Heliostat driver - this script should be run as a service

# In the end, this script should provide the remote IoT interface

# During development, it's invoked on command line

# Define some constants:

MOTOR_TRAVEL_TIME = 30  # seconds
AMOTOR = 1
HMOTOR = 2
EXTEND = 1
RETRACT = 2

from consolemenu import *
from consolemenu.items import *
import time



def calibrate():
    move_motor(AMOTOR, RETRACT, MOTOR_TRAVEL_TIME)
    move_motor(HMOTOR, RETRACT, MOTOR_TRAVEL_TIME)
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
        move_motor(HMOTOR, RETRACT, MOTOR_TRAVEL_TIME)


def scan():
    tcxs = []
    tcys = []
    tczs = []
    stop = time.time() + MOTOR_TRAVEL_TIME  # seconds
    start_moving_motor(HMOTOR, EXTEND, MOTOR_TRAVEL_TIME)
    while time.time() < stop:
        tcxs.append(...)
        tcys.append(...)
        tczs.append(...)
    sleep(0.1)  # seconds
    return tcxs, tcyx, tczs


def main():
    menu = ConsoleMenu("Heliostat", "Control Center")
    function_item = FunctionItem("Calibrate", calibrate)
    menu.append_item(function_item)
    menu.show()


if __name__ == "__main__":
    main()


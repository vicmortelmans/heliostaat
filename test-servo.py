#!/usr/bin/python
import RPi.GPIO as GPIO
import time
pin = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(pin, GPIO.OUT)
pwm = GPIO.PWM(pin, 500)
pwm.start(50)
while True:
    dc = int(input("Duty cycle: \n"))
    print(dc)
    pwm.ChangeDutyCycle(dc)

"""
pwm.start(99)
time.sleep(5)
for dc in range(99, -1, -1):
    print(dc)
    pwm.ChangeDutyCycle(dc)
    time.sleep(0.5)
"""
pwm.stop()
GPIO.cleanup()

#!/usr/bin/python
import pigpio
import time
gpio = pigpio.pi()
pinbcm = 18
gpio.set_PWM_frequency(pinbcm, 500)
gpio.set_PWM_range(pinbcm, 100)

gpio.set_PWM_dutycycle(pinbcm, 99)
time.sleep(10)
for dc in range(99, -1, -1):
    print(dc)
    gpio.set_PWM_dutycycle(pinbcm, dc)
    time.sleep(0.5)
gpio.stop()

#!/bin/bash
for i in {1..72}
do
  libcamera-still -o picture_$i.jpg
  timeout 7s feh --zoom 50 picture_$i.jpg
done

#!/usr/bin/python3
from bokeh.plotting import curdoc, figure
from bokeh.models import Button
import threading
import numpy as np
import time

handler_lock = threading.Semaphore(0)  # handler waiting for data
generator_lock = threading.Semaphore(0)  # generator waiting for user pushing the button

# THIS IS A BOKEH CALLBACK:
def data_handler():
    global data
    global title
    global final
    # wait until data is ready
    print("Handler: waiting for data from generator")
    handler_lock.acquire()
    print("Handler: handling data from generator")
    # add new graph
    p = figure(title=title)
    plot_circle = p.circle([],[])
    curdoc().add_root(p)
    plot_circle.data_source.stream(data)
    # add the button for the user to trigger the callback that will draw the next graph
    if not final:
        button = Button(label="Continue data generation", button_type="success")
        button.on_click(data_handler)
        curdoc().add_root(button)
    # unlock the generator to continue generating data
    print("Handler: unlocking generator")
    generator_lock.release()
    # actual drawing is only done after the callback finishes!

# THIS RUNS IN A SEPARATE THREAD:
def data_generation():
    global data
    global title
    global final
    final = False

    # Data generation phase 1 (starts immediately!)
    print("Generator: starting data generation")
    temp_x = np.random.rand(10)
    temp_y = np.random.rand(10)
    #time.sleep(10)  # pretending data generation takes long
    print("Generator: Data generated, going to plot")
    data = {'x': temp_x, 'y': temp_y}
    title = "graph 1"
    # unlock the bokeh callback that is waiting for data
    print("Generator: Unlocking handler")
    handler_lock.release()

    # lock the generator until the callback is done
    print("Generator: waiting for handler to finish")
    generator_lock.acquire()
    print("Generator: continuing generating data")
    # Data generation phase 2
    print("Generator: starting data generation")
    temp_x = np.random.rand(10)
    temp_y = np.random.rand(10)
    #time.sleep(10)  # pretending data generation takes long
    print("Data generated, going to plot")
    data = {'x': temp_x, 'y': temp_y}
    title = "graph 2"
    final = True  # no button needed to continue
    # unlock the bokeh callback that is waiting for data
    print("Generator: unlocking handler")
    handler_lock.release()


# start data generation in a separate thread
t = threading.Thread(target=data_generation)
t.start()

import pdb; pdb.set_trace()

# draw the start button
button = Button(label="Start handling data", button_type="success")
button.on_click(data_handler)
curdoc().add_root(button)

# main script finishes here, but the generator thread is still running and 
# iterative bokeh drawing is done in callbacks

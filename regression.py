from tkinter import *
import time
import random
import numpy as np
import embed_plot


WIDTH_IN_CELLS = 16
CELL_SIZE = 40

window = Tk()
window.resizable(False, False)

canvas = Canvas(window, 
                bg = "black", 
                highlightbackground = "blue",
                highlightthickness = 1,
                width = WIDTH_IN_CELLS * CELL_SIZE, 
                height = WIDTH_IN_CELLS * CELL_SIZE)

canvas.pack()

xTest = [i for i in range(100)]
yTest = [2*i for i in range(100)]

embed_plot.embedPlot(window, 2 * CELL_SIZE, 5 * CELL_SIZE, 3, xTest, yTest)

while True:
    window.update()

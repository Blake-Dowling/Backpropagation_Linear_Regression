from tkinter import *
import time
import random
import numpy as np
import embed_plot
import draw_network

##############################Constants##############################
WIDTH_IN_CELLS = 16
CELL_SIZE = 40
##############################Initialize Tkinter Window##############################
window = Tk()
window.resizable(False, False)
canvas = Canvas(window, 
                bg = "black", 
                highlightbackground = "blue",
                highlightthickness = 1,
                width = WIDTH_IN_CELLS * CELL_SIZE, 
                height = WIDTH_IN_CELLS * CELL_SIZE)

canvas.pack()
##############################Initialize Data##############################
inputVector = np.zeros(1000, ) #Initialize input vector
inputVector = np.array(list(map(lambda x : random.randint(0, 1000), inputVector))) #Fill 
#input vector with random integers 0-999

inputVectorReshaped = np.reshape(inputVector, (1000, )) #Reshape input vector into a column for
#vector multiplication
print(np.shape(inputVector))
randomGradient = random.randint(0, 1000) #Random correct gradient for f hat function

gradientVector = np.full(1000, randomGradient) #Initialize gradient array to be multiplied with input vector

fHatVector = np.multiply(inputVectorReshaped, gradientVector) #Expected output vector

gradientLabel = "Gradient: " + str(randomGradient) + "\n" #string to display actual gradient
canvas.create_text(6 * CELL_SIZE, 2.5 * CELL_SIZE, text = gradientLabel) #Display actual gradient

plot, plotCanvas = embed_plot.embedPlot(window, 2 * CELL_SIZE, 5 * CELL_SIZE, 3, inputVector, fHatVector) #
#Embed original plot containing correct data. Retrieve this plot in order to draw guess
#data in the plot.

##############################Guessing Weight##############################
guessWeight = 1
guessWeightLabel = "Guess Weight: " + str(guessWeight) + "\n" #string to guess gradient
canvas.create_text(6 * CELL_SIZE, 3.5 * CELL_SIZE, text = guessWeightLabel) #Display guess gradient



for input in inputVector:
    guessResult = input * guessWeight
    print(str(guessResult) + "\n")
    plot.plot([0, input], [0, guessResult], color = "red")
    plotCanvas.draw()
    #canvas.update()
    time.sleep(0.5)
    

#draw_network.displayData(canvas, 8, inputVector, "Input Data")
#xTrain = np.array([])



while True:
    window.update()

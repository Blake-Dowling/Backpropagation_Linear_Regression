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
inputVector = np.zeros(100, ) #Initialize input vector
inputVector = np.array(list(map(lambda x : random.randint(0, 100), inputVector))) #Fill 
#input vector with random integers 0-999
inputVectorReshaped = np.reshape(inputVector, (100, )) #Reshape input vector into a column for
#vector multiplication
randomGradient = random.randint(0, 100) #Random correct gradient for f hat function
gradientVector = np.full(100, randomGradient) #Initialize gradient array to be multiplied with input vector
yVector = np.multiply(inputVectorReshaped, gradientVector) #Expected output vector
gradientLabel = "Gradient: " + str(randomGradient) + "\n" #string to display actual gradient
canvas.create_text(6 * CELL_SIZE, 2.5 * CELL_SIZE, text = gradientLabel) #Display actual gradient
plot, plotCanvas = embed_plot.embedPlot(window, 2 * CELL_SIZE, 5 * CELL_SIZE, 3, inputVector, yVector) #
#Embed original plot containing correct data. Retrieve this plot in order to draw guess
#data in the plot.

##############################Guessing Weight##############################
weight = 1 #Final estimated regression
guessWeight = 0 #Algorithm's guess regression
guessWeightLabel = "Best Weight: " + str(weight) + "\n" #string to guess gradient
weightText = canvas.create_text(6 * CELL_SIZE, 3.5 * CELL_SIZE, text = guessWeightLabel) #Display guess gradient

meanSquaredError = 10000**2
meanSquaredErrorBest = 10000**2 + 1
mseText = canvas.create_text(6 * CELL_SIZE, 4.5 * CELL_SIZE, text = "Best Mean Squared Error: " + str(meanSquaredErrorBest)) #Display mean squared error


while meanSquaredError < meanSquaredErrorBest:
    meanSquaredErrorBest = meanSquaredError
    canvas.itemconfig(mseText, text = "Best Mean Squared Error: " + str(meanSquaredErrorBest))
    weight = guessWeight
    guessWeight = guessWeight + 1
    canvas.itemconfig(weightText, text = "Best Weight: " + str(weight)) #
    #Update guess weight label
    canvas.update()
    totalSquaredError = 0
    for i in range(len(inputVector)): #For each input
        #print(i)
        input = inputVector[i] #The current input iterated over
        actualResult = yVector[i] #The training data result
        guessResult = input * guessWeight #Algorithm guess
        squaredError = (guessResult - actualResult)**2 #Error between guess and training data
        totalSquaredError = totalSquaredError + squaredError #Cumulative error
        meanSquaredError = totalSquaredError / (i+1) #Mean error
        
        #canvas.itemconfig(mseText, text = "Mean Squared Error: " + str(meanSquaredError))
        #canvas.update_idletasks()

        #print(str(guessResult) + "\n")
        newLine, = plot.plot([0, input], [0, guessResult], color = "red") #Draw new line for current guess
        
        plotCanvas.draw()
        canvas.update()
        newLine.remove()
        #time.sleep(0.01)
print("Estimated Regression : " + str(weight))
#draw_network.displayData(canvas, 8, inputVector, "Input Data")
#xTrain = np.array([])



while True:
    window.update()

from tkinter import *
import time
import random
import numpy as np
import embed_plot
import draw_network

##############################Constants##############################
WIDTH_IN_CELLS = 16
CELL_SIZE = 40
HALF_SCREEN = (WIDTH_IN_CELLS / 4) * CELL_SIZE
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
##############################Initialize Input Data##############################
inputVector = np.zeros(10, ) #Initialize input vector
inputVector = np.array(list(map(lambda x : random.randint(0, 10), inputVector))) #Fill 
#input vector with random integers 0-999
inputVectorReshaped = np.reshape(inputVector, (10, )) #Reshape input vector into a column for
#vector multiplication
##############################Initialize Output Data##############################
randomGradient = random.randint(-10, 10) #Random correct gradient for y
randomBias = random.randint(0, 100) #Random correct bias for y
gradientVector = np.full(10, randomGradient) #Initialize gradient array to be multiplied with input vector
yVector = np.multiply(inputVectorReshaped, gradientVector) #Expected output vector
yVector = np.array(list(map(lambda x : x + randomBias, yVector))) #Expected output vector
gradientLabel = "Gradient: " + str(randomGradient) + "  Bias: " + str(randomBias) + "\n" #string to display actual gradient
canvas.create_text(HALF_SCREEN, 1 * CELL_SIZE, text = gradientLabel) #Display actual gradient
plot, plotCanvas = embed_plot.embedPlot(window, HALF_SCREEN, 8 * CELL_SIZE, 3, inputVector, yVector) #
#Embed original plot containing correct data. Retrieve this plot in order to draw guess
#data in the plot.
##############################Initialize Guess Data##############################
bias = random.randint(0, 100) #Initial guess bias
weight = random.randint(-10, 10) #Initial guess weight
guessLabel = "Best Weight: " + str(weight) + "  Best Bias: " + str(bias) + "\n" #string to guess gradient
guessText = canvas.create_text(HALF_SCREEN, 2 * CELL_SIZE, text = guessLabel) #Display guess gradient


mseBest = (100**2)*10 + 1
mseText = canvas.create_text(HALF_SCREEN, 3 * CELL_SIZE, text = "Best Mean Squared Error: " + str(mseBest)) #Display mean squared error
mseWI = canvas.create_text(HALF_SCREEN * 3, 1 * CELL_SIZE, text = "MSE Increase Weight: " + str(mseBest)) #Display MSE with weight increase
mseWD = canvas.create_text(HALF_SCREEN * 3, 2 * CELL_SIZE, text = "MSE Decrease Weight: " + str(mseBest)) #Display MSE with weight decrease
mseBI = canvas.create_text(HALF_SCREEN * 3, 3 * CELL_SIZE, text = "MSE Increase Bias: " + str(mseBest)) #Display MSE with bias increase
mseBD = canvas.create_text(HALF_SCREEN * 3, 4 * CELL_SIZE, text = "MSE Decrease Bias: " + str(mseBest)) #Display MSE with bias decrease

##############################Function for testing a guess parameter##############################
def testParams(weight, bias):
        totalSquaredError = 0
        mse = 100
        for i in range(len(inputVector)): #For each input
            ##############################Calculate Expected and Guess Values##############################
            input = inputVector[i] #The current input iterated over
            actualResult = yVector[i] #The training data result
            guessResult = (input * weight) + bias #Algorithm guess
            ##############################Calculate Error##############################
            squaredError = (guessResult - actualResult)**2 #Error between guess and training data
            totalSquaredError = totalSquaredError + squaredError #Cumulative error
            mse = totalSquaredError / (i+1) #Mean error
            ##############################Draw Guess Line##############################
            newPoint, = plot.plot(input, guessResult, "ro") #Draw new point for current guess
            newLine, = plot.plot([0, input, 10], 
                                 [0 + bias, guessResult, 10 * weight + bias], 
                                 color = "red") #Draw new line for current guess
            plotCanvas.draw()
            canvas.update()
            newPoint.remove()
            newLine.remove()
        return mse
##############################Main Guess Loop##############################
while True:
    ##############################Guess Parameters##############################
    mseWeightIncrease = testParams(weight + 1, bias)
    canvas.itemconfig(mseWI, text = "MSE Increase Weight: " + str(mseWeightIncrease)) #Display MSE with weight increase
    mseWeightDecrease = testParams(weight - 1, bias)
    canvas.itemconfig(mseWD, text = "MSE Decrease Weight: " + str(mseWeightDecrease)) #Display MSE with weight decrease
    mseBiasIncrease = testParams(weight, bias + 1)
    canvas.itemconfig(mseBI, text = "MSE Increase Bias: " + str(mseBiasIncrease)) #Display MSE with bias increase
    mseBiasDecrease = testParams(weight, bias - 1)
    canvas.itemconfig(mseBD, text = "MSE Decrease Bias: " + str(mseBiasDecrease)) #Display MSE with bias decrease
    canvas.update()
    ##############################Determine Best Guess##############################
    mseBestGuess = min(mseWeightIncrease, mseWeightDecrease, mseBiasIncrease, mseBiasDecrease)
    ##############################If MSE cannot be improved, stop program##############################
    if mseBestGuess >= mseBest:
        break
    ##############################Update Best MSE and Label##############################
    mseBest = mseBestGuess
    canvas.itemconfig(mseText, text = "Best Mean Squared Error: " + str(mseBest))
    canvas.update()
    ##############################Change optimal parameter##############################
    if mseWeightIncrease == mseBestGuess:
        weight = weight + 1
    elif mseWeightDecrease == mseBestGuess:
        weight = weight - 1
    elif mseBiasIncrease == mseBestGuess:
        bias = bias + 1
    elif mseBiasDecrease == mseBestGuess:
        bias = bias - 1
    ##############################Update Guess Label##############################
    guessLabel = "Best Weight: " + str(weight) + "  Best Bias: " + str(bias) + "\n" #string to guess gradient
    canvas.itemconfig(guessText, text = guessLabel) #
    canvas.update()
    #End guess loop


actualY = "Actual Y : " + str(randomGradient) + "x + " + str(randomBias)
estimatedY = "Estimated Y : " + str(weight) + "x + " + str(bias)
canvas.create_text(HALF_SCREEN, 4 * CELL_SIZE, text = actualY) #Display actual Y
canvas.create_text(HALF_SCREEN, 5 * CELL_SIZE, text = estimatedY) #Display estimated Y
#draw_network.displayData(canvas, 8, inputVector, "Input Data")
#xTrain = np.array([])



while True:
    window.update()

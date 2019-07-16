#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tkinter import *
import PIL
from PIL import Image, ImageTk, ImageGrab
from threading import Thread
import webbrowser
import numpy as np
from numpy import genfromtxt
import csv
import modules.NeuralNet as nn


#reading in the wheights and biases out of the data folder
wih = genfromtxt('data/wih.csv', delimiter=',')
who = genfromtxt('data/who.csv', delimiter=',')
bih = genfromtxt('data/bih.csv', delimiter=',')
bho = genfromtxt('data/bho.csv', delimiter=',')
bih_data=[]
for element in bih:
    bih_data.append([element])
bih=np.array(bih_data)
bho_data=[]
for element in bho:
    bho_data.append([element])
bho=np.array(bho_data)

 #########PAINTGUI##########

class Paint(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'white'
    def __init__(self):
        self.root = Tk()
        self.root.title("MINST Predictor")
        self.root.resizable(0,0)

        inputnodes=784
        hiddennodes=200
        outputnodes=10
        learningrate=0.1

        self.brain = nn.NeuralNet(inputnodes,hiddennodes,outputnodes,learningrate)
        self.brain.wih = wih
        self.brain.who = who
        self.brain.bias_ih = bih
        self.brain.bias_ho = bho

        self.brush_button = Button(self.root, text='Predict', command=self.Predict)
        self.brush_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='Clear', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.c = Canvas(self.root, bg='white', width=150, height=150,background="black")
        self.c.grid(row=1,columnspan=5)


        self.predictionLabel = Text(self.root, fg='blue', height=1, width=30,
                                            borderwidth=0, highlightthickness=0,
                                            relief='ridge')
        self.predictionLabel.grid(row=0, column=6)

        self.predictionScores = Text(self.root, height=10, width=30, padx=10,
                                        borderwidth=0, highlightthickness=0,
                                        relief='ridge')
        self.predictionScores.grid(row=1, column=6)

        self.image = Canvas(self.root, width=150, height=150,
                                highlightthickness=0, relief='ridge')
        self.image.create_image(0, 0, anchor=NW, tags="IMG")
        self.image.grid(row=2, rowspan=5, columnspan=5)
        #UI STUFF
        self.nnImageOriginal = Image.open("images/nn.png")
        self.resizeAndSetImage(self.nnImageOriginal)
        self.instagram = Label(self.root, text="@marvin_f_u_c_h_s", cursor="hand2")
        self.instagram.bind("<Button-1>", self.openInstagram)
        self.instagram.grid(row=4, column=6)
        self.github = Label(self.root, text="github.com/ApolloProgrammer", cursor="hand2")
        self.github.bind("<Button-1>", self.openGitHub)
        self.github1 = Label(self.root, text="Inspiration: github.com/salvacorts", cursor="hand2")
        self.github1.bind("<Button-1>", self.openGitHub1)
        self.github.grid(row=5, column=6)
        self.github1.grid(row=6, column=6)

        self.setup()
        self.root.mainloop()

    def openInstagram(self, event):
        webbrowser.open_new(r"https://instagram.com/marvin_f_u_c_h_s")

    def openGitHub(self, event):
        webbrowser.open_new(r"https://www.github.com/ApolloProgrammer")

    def openGitHub1(self, event):
        webbrowser.open_new(r"https://github.com/salvacorts/Keras-MNIST-Paint")

    def resizeAndSetImage(self, image):
        size = (150, 150)
        resized = image.resize(size, Image.ANTIALIAS)
        self.nnImage = ImageTk.PhotoImage(resized)
        self.image.delete("IMG")
        self.image.create_image(0, 0, image=self.nnImage, anchor=NW, tags="IMG")

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def Predict(self):
        self.save()
        img = Image.open("images/out.png")
        data = self.IMAGEtoDATA(img)
        scores=self.brain.feedforward(data)
        answer=np.argmax(scores)
        print(answer)
        self.predictionLabel.delete(1.0, END)
        self.predictionScores.delete(1.0, END)
        n = 0
        self.predictionLabel.insert(END, "This is a {}".format(answer))
        for score in scores:
            self.predictionScores.insert(END, "{}: {}\n".format(n, score))
            n += 1

    def save(self):
        x = self.root.winfo_rootx() + 11
        y = self.root.winfo_rooty() + 100
        x1 = x + 300
        y1 = y +300
        ImageGrab.grab((x,y, x1, y1)).save("images/out.png")

    def IMAGEtoDATA(self,img):
        size=28,28
        img.thumbnail(size, Image.ANTIALIAS)
        img.save("images/out.png", "PNG")
        iar = np.asarray(img)
        array = np.zeros((784, 1))
        i = 0
        for row in iar:
            for pixel in row:
                array[i] = pixel[0] / 255
                i += 1
        data = []
        for element in array:
            data.append(element[0])
        i=0
        for element in data:
            if element!=0.0:
                data[i]=element*255
            else:
                data[i]=0
            i+=1
        print(data)
        return data

    def use_eraser(self):
        self.predictionLabel.delete(1.0, END)
        self.predictionScores.delete(1.0, END)
        self.c.delete("all")
        self.resizeAndSetImage(self.nnImageOriginal)

    def paint(self, event):
        self.line_width = self.DEFAULT_PEN_SIZE
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()

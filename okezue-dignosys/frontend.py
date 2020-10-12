import backend



from TkinterDnD2 import *
try:
    from Tkinter import *
except ImportError:
    from tkinter import *


from tkinter import Frame, Tk, BOTH, Label, Menu, filedialog, messagebox, Text
from PIL import Image, ImageTk

import os
import codecs

screenWidth = "1450"
screenHeight = "800"
windowTitle = "[Okezue Bell] Dignosys: Respiratory Disease Detection with X-Rays"

import cv2
from random import randint
class Window(Frame):
    _PRIOR_IMAGE = None
    

    DIAGNOSIS_RESULT = ""
    DIAGNOSIS_RESULT_FIELD = None
 



    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        
        load = Image.open("fusion-medical-animation-EAgGqOiDDMg-unsplash-scaled.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=(int(screenWidth)/2)-load.width/2, y=((int(screenHeight)/2))-load.height/2-80)
        self._PRIOR_IMAGE = img 
        self.DIAGNOSIS_RESULT_FIELD = Text(self,  width=int(screenWidth), height=13)
        self.DIAGNOSIS_RESULT_FIELD.pack ( )
        self.DIAGNOSIS_RESULT_FIELD.place(x=0, y=int(screenHeight)-200)
        
        
    def addDiagnosisResult (self, value):
        self.DIAGNOSIS_RESULT_FIELD.delete("1.0","end") 
        self.DIAGNOSIS_RESULT = "" 
        self.DIAGNOSIS_RESULT_FIELD.insert(1.0, value)
        
        
                                          
root = Tk()
app = Window(root)




root.wm_title(windowTitle)
root.geometry(screenWidth + "x" + screenHeight)


menubar = Menu(root)


filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Files", menu=filemenu)

CONSTANT_DIAGNOSIS_IMAGE_SPAN = 480


def loadRegularPneumoniaImageFromDialog():
    currdir = os.getcwd()
    image_file = filedialog.askopenfile(mode ='r', parent=root, initialdir=currdir, title='Please select an Xray Image of suspected regular pneumonia case:')
    root.wm_title(windowTitle + " : " + image_file.name)
    loadRegularPneumoniaImageFromName(image_file.name)

def loadRegularPneumoniaImageFromName(filename):
    app._PRIOR_IMAGE.destroy() 
    load = Image.open(filename)
    load = load.resize((CONSTANT_DIAGNOSIS_IMAGE_SPAN, CONSTANT_DIAGNOSIS_IMAGE_SPAN),Image.ANTIALIAS) 
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=(int(screenWidth)/2)-CONSTANT_DIAGNOSIS_IMAGE_SPAN/2, y=((int(screenHeight)/2))-CONSTANT_DIAGNOSIS_IMAGE_SPAN/2-80)
    app.DIAGNOSIS_RESULT += "**Pneumonia Mode Result**\n" + filename+"\n\n"
    app.DIAGNOSIS_RESULT += backend.func_regularPneumonia (filename)
    print(app.DIAGNOSIS_RESULT)
    app._PRIOR_IMAGE = img 
    app.addDiagnosisResult(app.DIAGNOSIS_RESULT)
    enableDiagnosisResultColouring ( )
    
def loadCovid19ImageFromDialog():
    currdir = os.getcwd()
    image_file = filedialog.askopenfile(mode ='r', parent=root, initialdir=currdir, title='Please select an Xray Image of suspected coronavirus case:')
    root.wm_title(windowTitle + " : " + image_file.name)
    loadCovid19ImageFromName(image_file.name)

def loadCovid19ImageFromName(filename):
    app._PRIOR_IMAGE.destroy()
    load = Image.open(filename)
    load = load.resize((CONSTANT_DIAGNOSIS_IMAGE_SPAN, CONSTANT_DIAGNOSIS_IMAGE_SPAN),Image.ANTIALIAS) 
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=(int(screenWidth)/2)-CONSTANT_DIAGNOSIS_IMAGE_SPAN/2, y=((int(screenHeight)/2))-CONSTANT_DIAGNOSIS_IMAGE_SPAN/2-80)
    app.DIAGNOSIS_RESULT +=  "**Covid19 Mode Result**\n" + filename+"\n\n"
    app.DIAGNOSIS_RESULT += backend.func_covid19Pneumonia (filename)
    print(app.DIAGNOSIS_RESULT)
    app._PRIOR_IMAGE = img 
    app.addDiagnosisResult(app.DIAGNOSIS_RESULT)
    enableDiagnosisResultColouring ( )

def loadlungcancerImageFromDialog():
    currdir = os.getcwd()
    image_file = filedialog.askopenfile(mode ='r', parent=root, initialdir=currdir, title='Please select an Xray Image of suspected lung cancer case:')
    root.wm_title(windowTitle + " : " + image_file.name)
    loadCovid19ImageFromName(image_file.name)

def loadlungcancerImageFromName(filename):
    app._PRIOR_IMAGE.destroy()
    load = Image.open(filename)
    load = load.resize((CONSTANT_DIAGNOSIS_IMAGE_SPAN, CONSTANT_DIAGNOSIS_IMAGE_SPAN),Image.ANTIALIAS) 
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=(int(screenWidth)/2)-CONSTANT_DIAGNOSIS_IMAGE_SPAN/2, y=((int(screenHeight)/2))-CONSTANT_DIAGNOSIS_IMAGE_SPAN/2-80)
    app.DIAGNOSIS_RESULT +=  "**Lung Cancer Mode Result**\n" + filename+"\n\n"
    app.DIAGNOSIS_RESULT += backend.func_covid19Pneumonia (filename)
    print(app.DIAGNOSIS_RESULT)
    app._PRIOR_IMAGE = img 
    app.addDiagnosisResult(app.DIAGNOSIS_RESULT)
    enableDiagnosisResultColouring ( )


    
filemenu.add_command(label="Load image to test for pneumonia", command=loadRegularPneumoniaImageFromDialog)
filemenu.add_command(label="Load image to test for covid-19", command=loadCovid19ImageFromDialog)
filemenu.add_command(label="Load image to test for lung cancer", command=loadlungcancerImageFromDialog)

def colourDiagnosisMessageText ( diagnosisContent, startIndexText, endIndexText ):
 
    if ( backend.DIAGNOSE[0] in diagnosisContent or backend.DIAGNOSE[1] in diagnosisContent ):
        app.DIAGNOSIS_RESULT_FIELD.tag_add("DIAGNOSIS_RESULT_MESSAGE", startIndexText, endIndexText)
        app.DIAGNOSIS_RESULT_FIELD.tag_configure("DIAGNOSIS_RESULT_MESSAGE", background="red", foreground ="white")


    if ( backend.DIAGNOSE[2] in diagnosisContent ):
        app.DIAGNOSIS_RESULT_FIELD.tag_add("DIAGNOSIS_RESULT_MESSAGE", startIndexText, endIndexText)
        app.DIAGNOSIS_RESULT_FIELD.tag_configure("DIAGNOSIS_RESULT_MESSAGE", background="green", foreground ="white")
        

def enableDiagnosisResultColouring ( ):
    diagnosisResultFieldContent = app.DIAGNOSIS_RESULT_FIELD.get("1.0","end")
    colourDiagnosisMessageText ( diagnosisResultFieldContent, "4.0", "4.21" )


root.config(menu=menubar)
root.mainloop()

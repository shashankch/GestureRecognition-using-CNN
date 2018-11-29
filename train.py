import sys
sys.path.append('D:/MINOR PROJECT')
import Matrix_CV_ML3D as DImage
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')



x= DImage.Matrix_CV_ML3D('D:/MINOR PROJECT/train',65,50)
x.build_ML_matrix()

#G:\Deep Learning\HandGestures\train
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import pandas as pd

y=np_utils.to_categorical(x.labels)
x=x.global_matrix
x=x.astype('float32')/255
print(x.shape)

model= Sequential()


model.add(Convolution2D(32,3,3,activation='relu',input_shape=(3,50,65)))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,batch_size=32,nb_epoch=10,verbose=2)

print(model.summary())
score=model.evaluate(x,y,verbose=1)

test = DImage.Matrix_CV_ML3D('D:/MINOR PROJECT/test2',65,50)
test.build_ML_matrix()

labels=test.labels
y = np_utils.to_categorical(test.labels)
test = test.global_matrix
test = test.astype('float32')/255

score = model.evaluate(test, y, verbose=1)
pred  = model.predict(test)
predicted = np.argmax(pred,axis=1)
print(score)
labels=pd.Series(labels).to_frame('Labels')
pred = pd.Series(predicted)
predicted=pd.Series(predicted).to_frame('Predicted')
s = np.array(pred)
p=pd.concat((labels,predicted),axis=1)
print(p)
print("")
message = ""
for x in np.nditer(s):
    if(x==0):
        message += "H"
    elif(x==1):
        message += "E"
    elif(x==2):
        message += "L"
    elif(x==3):
        message += "O"

print("The decoded message is " + message)
model.save('my_model.h5')
import sys
from tkinter import Button,Label,Tk
from tkinter import filedialog
import shutil
import os
from PIL import Image,ImageTk
sys.path.append('D:/MINOR PROJECT')
import Matrix_CV_ML3D as DImage
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')


root = Tk()
root.title("Join")
root.geometry("300x300")
root.configure(background='grey')
root.title("HAND GESTURE RECOGNITION")
label_welcome=Label(font=('arial',18,"bold italic"),text="Welcome to Hand Gesture Recognition!!", fg="Purple")
label_image=Label(root)

root = Tk()
root.geometry("1000x500+0+0")
root.title("HAND GESTURE RECOGNITION")
label_welcome=Label(font=('arial',18,"bold italic"),text="Welcome to Hand Gesture Recognition!!", fg="Purple")
label_image=Label(root)#label used for canvas in line 118

root.filename =  filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
root.filename = root.filename.replace(r'/',r'//')
path=root.filename

basewidth = 300
img = Image.open(path)
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
img.save(path)

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img = ImageTk.PhotoImage(Image.open(path))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
panel = Label(image = img)
panel.grid(row=5, column=1)
#The Pack geometry manager packs widgets in rows or columns.

def chooseFile():
    global panel
    root.filename =  filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    root.filename = root.filename.replace(r'/',r'//')
    path=root.filename
    basewidth = 300
    img = Image.open(path)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(path)
    panel.destroy()
    panel = Label(image = img)
    panel.grid(row=5, column=1)
    #panel.configure(image = img)

label_row2=Label(text="")
label_row3=Label(text="")
label_chooseFile=Label(font=('arial',12),text="Select a file as input::")
b1=Button(padx=16,pady=16,fg="blue", text="Choose File",command=lambda: chooseFile(),cursor="target")
b2=Button(padx=16,pady=16,fg="blue", text="Meaning?",command=lambda: predictor())
label_predictedValue=Label(font=('arial',14),text="")
label_row13=Label()
label_row11=Label()

label_welcome.grid(row=0,column=1)
label_image.grid(row=2,column=1)
label_row2.grid(row=2,column=0)
label_row3.grid(row=3,column=0)
label_chooseFile.grid(row=10,column=0)
b1.grid(row=10,column=1)
label_row11.grid(row=11,column=0)
label_predictedValue.grid(row=12,column=1)
label_row13.grid(row=13,column=0)
b2.grid(row=14,column=1)

    def build_ML_matrix(self):
        counter = 0
        self.labels  = np.empty([len(self.onlyfiles)],dtype="int32")
        for file in self.onlyfiles:
            q = self.prepare_matrix(file)
            if (counter != 0):
                print(str(q.shape) + str(self.global_matrix.shape))
                self.global_matrix = np.concatenate([self.global_matrix,q],axis=0)
            else:
                self.global_matrix = q
            dash          = file.rfind("_")
            dot           = file.rfind(".")
            classtype     = file[dash+1:dot]
            self.labels[counter] = classtype
            counter = counter + 1

import os
from PIL import Image,ImageTk
sys.path.append('D:/MINOR PROJECT')
import Matrix_CV_ML3D as DImage
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')



x= DImage.Matrix_CV_ML3D('D:/MINOR PROJECT/HandGestures/train',65,50)
x.build_ML_matrix()

#G:\Deep Learning\HandGestures\train
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Activation
from keras.layers import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD
#from keras.datasets import cifar10
import pandas as pd

y=np_utils.to_categorical(x.labels)
x=x.global_matrix
x=x.astype('float32')/255
print(x.shape)

model= Sequential()


model.add(Convolution2D(32,3,3,activation='relu',input_shape=(3,50,65)))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,batch_size=32,nb_epoch=1,verbose=1)

print(model.summary())
score=model.evaluate(x,y,verbose=1)

def predictor():
    test = DImage.Matrix_CV_ML3D('D:/MINOR PROJECT/HandGestures/test1',65,50)
    test.build_ML_matrix()

    labels=test.labels
    y = np_utils.to_categorical(test.labels)
    test = test.global_matrix
    test = test.astype('float32')/255

    score = model.evaluate(test, y, verbose=1)
    pred  = model.predict(test)
    predicted = np.argmax(pred,axis=1)
    print(score)
    labels=pd.Series(labels).to_frame('Labels')
    pred = pd.Series(predicted)
    predicted=pd.Series(predicted).to_frame('Predicted')
    s = np.array(pred)
    p=pd.concat((labels,predicted),axis=1)
    print(p)
    print(s[0])
    string = str(s[0])
    str1 = "The GESTURE is a ::" + string
    label_predictedValue.configure(text=str1)

def selectFile(str):
    print(str)
    str1 = str[:-9] + "1" + str[-9:]
    folder = str1[:-9]
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    shutil.copyfile(str,str1)

root = Tk()
root.geometry("600x200+0+0")
root.title("HAND GESTURE RECOGNITION")
label_welcome=Label(font=('arial',18,"bold italic"),text="Welcome to Hand Gesture Recognition!!", fg="Purple")
label_image=Label(root)#label used for canvas in line 118

def chooseFile():
    root.filename =  filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    selectFile(root.filename)
    root.filename = root.filename.replace(r'/',r'//')

    img = Image.open(root.filename)
    render = ImageTk.PhotoImage(img)
    canvas = Canvas(label_image,height=100,width=100)
    canvas.image=render
    canvas.create_image(0,0,anchor='center',image=render)
    canvas.pack()

label_row2=Label(text="")
label_row3=Label(text="")
label_chooseFile=Label(font=('arial',12),text="Select a file as input::")
b1=Button(padx=16,pady=16,fg="blue", text="Choose File",command=lambda: chooseFile(),cursor="target")
b2=Button(padx=16,pady=16,fg="blue", text="Meaning?",command=lambda: predictor())
label_predictedValue=Label(font=('arial',14),text="")
label_row13=Label()
label_row11=Label()

label_welcome.grid(row=0,column=1)
label_image.grid(row=2,column=1)
label_row2.grid(row=2,column=0)
label_row3.grid(row=3,column=0)
label_chooseFile.grid(row=10,column=0)
b1.grid(row=10,column=1)
label_row11.grid(row=11,column=0)
label_predictedValue.grid(row=12,column=1)
label_row13.grid(row=13,column=0)
b2.grid(row=14,column=1)

root.mainloop()

#Start the GUI
root.mainloop()

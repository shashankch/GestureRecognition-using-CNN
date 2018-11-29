import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
class Matrix_CV_ML3D:

    def __init__(self,path,height,width):
        if not path.endswith('/'):
            path           = path + "/"
        self.path      = path
        self.height    = height
        self.width     = width
        self.type      = type
        self.onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]


    def prepare_matrix(self,pathx):
        path          = self.path + pathx
        img           = cv2.imread(path)
        img           = cv2.resize(img,(self.width,self.height), interpolation = cv2.INTER_CUBIC)
        img = np.reshape(img,(1,3,self.width,self.height))
        return (img)

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

import ast

from characterDetector import readCharacters, showCharacters
from classifier import *
import cv2 as cv
from croppTool import croppImage
from simpy import *
images, boxes = readCharacters(cv.imread("examples/IMG_20201218_142508.jpg"))

string = ""
for i in images:
    label = classify(i)

    if label == "x":
        string = string + "*"
        continue

    if label == "rbr":
        label = ")"

    elif label == "lbr":
        label = "("

    elif label== "div":
        label = "/"

    string = string + label
    string = "7-8"

print(eval(string))
expresion= ""
first=True
eq = ()


for i in range(len(string)):
    if string[i] =="0" or string[i] == "1" or string[i] == "2" or string[i] == "3" or string[i] == "4" or string[i] == "5" or string[i] == "6" or string[i] == "7" or string[i] == "8" or string[i] == "9":
        if first==False:
            expresion = 10*int(expresion) + int(string[i])
        else:
            expresion = int(string[i])
            first = False


    if string[i] == "+" or string[i] == "-" or ord(string[i]) ==42 or string[i] == "/" or string[i] == "(" or string[i] == ")":
        if len(eq)==0:
            eq = (int(expresion), string[i])
        else:
            eq=(*eq, expresion)
            eq=(*eq, string[i])
        expresion = ""
        first=True


    if i==len(string)-1:
        eq = (*eq, expresion)

#print(eq)









#     string = "2+3"
#
#     for i in range(3):
#         if string[i] == "+":
#             x=string.split("+")
#             eq = int(x[0]) + int(x[1])
#
# print(eq)







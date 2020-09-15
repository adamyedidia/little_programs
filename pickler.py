import pickle
from PIL import Image
import math

print "Salut ma cherie"
print "What is the password?"
PASSWORD = raw_input("--> ")

print "Where is the image?"
IMAGE_PATH = raw_input("--> ")

imageData = Image.open(IMAGE_PATH)
width, height = imageData.size
pixelData = list(imageData.getdata())

#print pixelData

def convertTextToNumber(text):
    return int(''.join(str(ord(c)) for c in text))



newPixelData = pixelData

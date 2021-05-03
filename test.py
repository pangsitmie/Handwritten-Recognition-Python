import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from PIL import Image, ImageDraw, ImageFont
import time
from keras.utils import plot_model

start = time.time()

#Define Path
model_path = './model.h5'
test_path = './data/test'
filename = r'C:\Users\jerie\OneDrive\Desktop\handwritting fix\data\test\3.png'

#Load the pre-trained models
model = load_model(model_path)
img_width, img_height = 150,150

#Prediction Function
def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x) / 255.0
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    #array_p = model.predict_proba(x,verbose=1)
    result = array[0]
    #percentage = array_p[0]
    answer = np.argmax(result,axis=-1)
    #print(result[answer][0])

    if answer <10:
        print("Predicted: ", answer)
        pred_class=answer
    else:
        if answer == 10:
            print("Predicted: A")
            pred_class = "A"
        elif answer == 11:
            print("Predicted: B")
            pred_class = "B"
        elif answer == 12:
            print("Predicted: C")
            pred_class = "C"
        elif answer == 13:
            print("Predicted: D")
            pred_class = "D"
        elif answer == 14:
            print("Predicted: E")
            pred_class = "E"
        elif answer == 15:
            print("Predicted: F")
            pred_class = "F"
        elif answer == 16:
            print("Predicted: G")
            pred_class = "G"
        elif answer == 17:
            print("Predicted: H")
            pred_class = "H"
        elif answer == 18:
            print("Predicted: I")
            pred_class = "I"
        elif answer == 19:
            print("Predicted: J")
            pred_class = "J"

    print(array[0:])
    #print(percentage[0:3])
    img = Image.open(file)
    img_height_sc = int((float(img.size[1])*float(img_width/float(img.size[0]))))
    img = img.resize((img_width,img_height_sc))
    draw = ImageDraw.Draw(img)
    draw.rectangle(((0,img_height_sc),(200,img_height_sc)),fill = (255,255,255))
    fnt = ImageFont.truetype("times.ttf", 17)
    draw.multiline_text((1,img_height_sc - 20),"Prediction: "+pred_class, font=fnt,fill = (0,0,0))
    del draw

    img.show()
    return answer


#Walk the directory for every image
"""for i, ret in enumerate(os.walk(test_path)):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
    
    print(ret[0] + '/' + filename)
    result = predict(ret[0] + '/' + filename)
    print(" ")"""
result = predict(filename)



print(" ")
#Calculate execution time
end = time.time()
dur = end-start

if dur<3600:
    print("Execution Time:",dur,"seconds")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
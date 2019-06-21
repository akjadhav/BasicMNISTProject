from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def preprocess(photo_path):
    img = Image.open(photo_path)
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.asarray(img, dtype="int32")
    fig = plt.imshow(img, cmap = 'gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    img = img.reshape(-1, 28, 28, 1) / 255
    
    return img

model = load_model(r'BasicMNISTModel.h5')

photo_path = r'Prediction Data\5.png' # file path of the image that is going to be predicted
image = preprocess(photo_path)

print('Model Predicted:' , int(model.predict_classes(image)))

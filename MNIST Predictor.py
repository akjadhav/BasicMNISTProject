
# coding: utf-8

# In[1]:


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def preprocess(photo_path):
    img = Image.open(photo_path)
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.asarray(img, dtype="int32")
    plt.imshow(img, cmap = 'gray')
    img = img.reshape(-1, 28, 28, 1) / 255
    
    return img


# In[10]:


model = load_model(r'BasicMNISTModel.h5')

photo_path = r'Prediction Data\two.jpeg'
image = preprocess(photo_path)

print('Model Predicted:' , int(model.predict_classes(image)))


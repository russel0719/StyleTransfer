import numpy as np
from PIL import Image
from IPython.display import display 

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    display(img)
    img.save(filename)
import sys, os
sys.path.append(os.pardir)
import numpy as np
from PIL import Image

def img_show(img, isReshape=True):
    if isReshape:
        img = np.reshape(img, (28, 28))
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
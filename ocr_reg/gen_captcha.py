import random
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import os

random.seed(42)


def random_captcha():
    captcha_text = []
    for i in range(4):
        c = random.choice('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        captcha_text.append(c)
    return ''.join(captcha_text)


for i in range(10):
    print(random_captcha())


def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image


count = 5000
path = '/home/wq/kaggle/ext-data/captcha_ext'
if not os.path.exists(path):
    os.makedirs(path)
for i in range(count):
    text, image = gen_captcha_text_and_image()
    filename = text + '.png'
    image.save(path + os.path.sep + filename)
    print('saved %d : %s' % (i+1, filename))

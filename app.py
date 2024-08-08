import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('climbing.pkl')

categories = ('Brick wall', 'Climbing wall')

def classify_image(image):
    #img_resized = resize_image_pil(image, 224, 224)
    pred,idx,probs = learn.predict(image)
    return dict(zip(categories, map(float,probs)))

image = gr.Image()
label = gr.Label()
examples = ['climbing.jpg', 'brick.jpg']

title = "Wall Classifier"
intf = gr.Interface(fn=classify_image, inputs = image, outputs=label, examples = examples, allow_flagging="never", title = title)
intf.launch(inline=False)
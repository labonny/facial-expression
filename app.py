import gradio as gr
from fastai.vision.all import *
import cv2

learn = load_learner('model-v4.pkl')
labels = learn.dls.vocab
def predict(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, dstCn=3 )
    pred,pred_idx,probs = learn.predict(image)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Facial Expression Classifier"
description = "A facial expression classifier, trained using the <a href='https://www.kaggle.com/datasets/msambare/fer2013'>FER-2013 dataset</a>. This dataset consists of 28,709 examples of faces: each one is 48x48 grayscale pixels and is labelled with one of the following expressions: anger, disgust, fear, happy, neutral, sad, surprise.<p><p>This was used to train a resnet34 model."
examples = ["angryExample.jpg", "disgustExample.jpg", "fearExample.jpg", "happyExample.jpg", "neutralExample.jpg", "sadExample.jpg", "surpriseExample.jpg"]
iface = gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(48,48)), outputs=gr.outputs.Label(num_top_classes=3), examples=examples, title=title, description=description,interpretation='default')
iface.launch()


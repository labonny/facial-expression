import gradio as gr
from fastai.vision.all import *
import cv2
import PIL

learn = load_learner('fec224-resnet34-v1.pkl')
labels = learn.dls.vocab
def predict(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, dstCn=3 )
    pred,pred_idx,probs = learn.predict(image)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Facial Expression Classifier"
description = "A facial expression classifier, trained using the <a href='https://www.kaggle.com/datasets/msambare/fer2013'>FER-2013 dataset</a>. This dataset consists of 28,709 examples of faces: each one is 48x48 grayscale pixels and is labelled with one of the following expressions: anger, disgust, fear, happy, neutral, sad, surprise.<p><p>This was used to train a resnet34 model: the code to train the model is available as a <a href='https://www.kaggle.com/code/l4bonny/facialexpressionclassifier-model'>kaggle notebook</a>. The model has an error rate of about 31% on the validation set.<p>The model expects just the face, with everything else cropped out (see the sample test set images below). You can use one of the sample images below, taken from the test set of the FER-2013 dataset, or you can crop your own image after uploading it: click on the pencil icon and then drag the edges of the box to crop, and drag the inside of the box to recenter."
examples = ["angryExample.jpg", "disgustExample.jpg", "fearExample.jpg", "happyExample.jpg", "neutralExample.jpg", "sadExample.jpg", "surpriseExample.jpg"]
iface = gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(224,224)), outputs=gr.outputs.Label(num_top_classes=3), examples=examples, title=title, description=description,interpretation='default')
iface.launch()


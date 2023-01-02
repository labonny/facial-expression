import gradio as gr
from fastai.vision.all import *

learn = load_learner('model-v4.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

iface = gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(48,48)), outputs=gr.outputs.Label(num_top_classes=3))
iface.launch()


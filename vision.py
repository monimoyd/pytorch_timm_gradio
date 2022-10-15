import urllib

import gradio as gr
import torch
import timm
import numpy as np

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from typing import Dict
import boto3

AWS_ACCESS_KEY='<AWS ACCESS KEY>'
AWS_SECRET_KEY='<AWS ACCESS KEY>'
BUCKET_NAME='<AWS BUCKET>'
MODEL_PATH='model.script.pt'

# Download the model from S3 bucket

s3 = boto3.resource('s3', aws_access_key_id = AWS_ACCESS_KEY, aws_secret_access_key = AWS_SECRET_KEY)
	
bucket = s3.Bucket(BUCKET_NAME)
bucket.download_file( MODEL_PATH, MODEL_PATH )
print(" Model downloaded")
  
model = torch.jit.load(MODEL_PATH)
print(" Model loaded")

# Download human-readable labels for CIFAR10.
# get the classnames
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image: Image) -> Dict[str, float]:

    if image is None:
        return None

    image = image.resize((224, 224), Image.BILINEAR)
    image = np.array(image)   

    image = torch.tensor(image[None, None, ...], dtype=torch.float32)
    image = image.squeeze(0)
    image = image.permute(0,3,1,2)
    preds = model.forward_jit(image)

    return {cifar10_labels[i]: float(preds[i]) for i in range(10)}

if __name__ == "__main__":
    gr.Interface(
        fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=10)
    ).launch(server_name="0.0.0.0")
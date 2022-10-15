# pytorch_timm_gradio

## How to run build and run container

### Step1:
Upload teh trained model to the AWS bucker

### Step2:
Replace the <AWS_ACCESS_KEY>, <AWS_SECRET_KEY>, <AWS BUCKET> in file vision.py with actual values

### Step3:
Build container using the command:

docker build -t timm-gradio .

### Step4:
Run container using the command:

docker run -p 8080:7860 timm-gradio

(Make sure that port 8080 is open in the host)


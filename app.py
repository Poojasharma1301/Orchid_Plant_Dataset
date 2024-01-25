from flask import Flask, render_template, request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

img_size=256

app = Flask(__name__) 

model=load_model('model.hdf5')

label_dict={0:'cattleya', 1:'dendrobium', 2:'oncidium', 3:'phalaenopsis',4:'vanda'}

def preprocess(img):

	img=np.array(img)

	if(img.ndim==4):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size,3)
	return reshaped

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)
	#test_img = np.asarray(test_image)
	prediction = model.predict(test_image)
	print(prediction)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])
	print(prediction,result,accuracy)
	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return response

app.run(debug=True)
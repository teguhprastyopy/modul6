from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


app = Flask(__name__)

dic = {0: 'paper', 1: 'rock', 2: 'scissors'}

model = load_model('RPS-MODUL5.h5')

model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(224,224))
	i = img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	p = model.predict(i)
	index = np.argmax(p, axis=-1)
	return dic[index[0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	filenames =['static/output.png', 'static/output1.png', 'static/output2.png']
	return render_template("about.html", filenames=filenames)

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
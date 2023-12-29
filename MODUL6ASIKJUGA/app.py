from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import time
import numpy as np
import os
import random


app = Flask(__name__)

dic = {0: 'paper', 1: 'rock', 2: 'scissors'}

model = load_model('RPS-MODUL5.h5')

model.make_predict_function()

def load_test_dataset(directory, target_size=(224, 224), num_classes=3):
    images = []
    labels = []

    label_to_int = {'paper': 0, 'rock': 1, 'scissors': 2}
    
    for label_dir in os.listdir(directory):
        class_dir = os.path.join(directory, label_dir)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = load_img(img_path, target_size=target_size)
                img = img_to_array(img)
                img /= 255.0  # Scale pixel values
                images.append(img)
                labels.append(label_to_int[label_dir])  # Convert labels here

    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=num_classes)

    return images, labels

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	filenames =['static/output.png', 'static/output1.png', 'static/output2.png']
	return render_template("about.html", filenames=filenames)

@app.route("/choose-image", methods=['GET'])
def choose_image():
    # Misalkan gambar disimpan dalam folder 'static/images'
    image_folder = 'static/images'
    image_files = os.listdir(image_folder)
    # Pilih secara acak 25 gambar
    selected_images = random.sample(image_files, 25)
    return render_template("choose_image.html", images=selected_images)

def predict_label(img_path):
    start_time = time.time()
    i = load_img(img_path, target_size=(224,224))
    i = img_to_array(i)/255.0
    i = i.reshape(1, 224,224,3)
    p = model.predict(i)
    end_time = time.time()
    prediction_time = end_time - start_time
    index = np.argmax(p, axis=-1)
    return dic[index[0]], prediction_time

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename    
        img.save(img_path)

        prediction, prediction_time = predict_label(img_path)

        # Load your test dataset
        # You should specify the path to your test dataset directory here
        test_images, test_labels = load_test_dataset("test_dataset", num_classes=3)

        # Evaluate the model
        loss, accuracy = model.evaluate(test_images, test_labels)

        accuracy = round(accuracy * 100, 1)
        return render_template("index.html", 
                               prediction=prediction, 
                               accuracy=accuracy, 
                               prediction_time=prediction_time, 
                               img_path=img_path, 
                               loss=loss)

@app.route("/predict", methods=['GET'])
def predict():
    image_filename = request.args.get('image', '')  # Get the image filename from query parameters
    if not image_filename:
        return "No image specified.", 400  # Return a bad request error if no image is specified

    img_path = os.path.join('static/images', image_filename)  # Construct the full image path
    if not os.path.exists(img_path):
        return "Image not found.", 404  # Return a not found error if the image does not exist

    prediction, prediction_time = predict_label(img_path)
    return render_template("prediction.html", prediction=prediction, prediction_time=prediction_time, img_path=img_path)

if __name__ =='__main__':
	#app.debug = True
    app.run(host='0.0.0.0', debug = True)
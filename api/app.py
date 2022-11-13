from flask import Flask, request
from joblib import load
import os
from PIL import Image
from numpy import asarray
# from keras.preprocessing import image
import numpy as np



app = Flask(__name__)
app.secret_key = "super secret key"

model_path = "svm_gamma=0.004_C=0.8.joblib"
model = load(model_path)

@app.route('/')
def hello():
    return "hello"


@app.route('/uploadFile', methods=['GET', 'POST'])
def upload_file():
    try:
        print("In Upload File")
        if request.method == 'POST' and 'file' in request.files and request.files['file'].filename != '':
          print("File Name is ")
          print(request.files['file'].filename)
          request.files['file'].save(os.path.join('uploaded_files', request.files['file'].filename))
          
          img = Image.open('uploaded_files/image.png')
          
        #   img = img.reshape(8, 8)
          img=img.resize((8,8))
          
        #   img = arr.reshape(8, 8)
          img = np.array(img).reshape((-1, 64))
          print("Arr")
          print(img)
        #   img = list(arr)
          print(img)

          print("Image error")
        #   print(img)

          predicted = model.predict(img)
          return {"y_predicted":int(predicted[0])}
        else:
          return "Use Method Post and Upload File by selecting form-data and key name as file with type as file"
    except Exception as e:
        print(e)
        return "Error Received"

@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    print("done loading")
    print(image)
    print("--------")
    print(np.array(image).shape)
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}

if __name__ == "__main__":
    app.debug = True
    app.run()

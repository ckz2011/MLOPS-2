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
        if request.method == 'POST' and 'file1' in request.files and request.files['file1'].filename != '':
          
          print(request.files['file1'].filename)
          print(request.files['file2'].filename)

          request.files['file1'].save(os.path.join('uploaded_files', request.files['file1'].filename))
          request.files['file2'].save(os.path.join('uploaded_files', request.files['file2'].filename))
          
          img1 = Image.open("uploaded_files/{}".format(request.files['file1'].filename))
          img2 = Image.open("uploaded_files/{}".format(request.files['file2'].filename))

          print("Images Fetched")
          
        #   img = img.reshape(8, 8)
        #   img=img.resize((8,8))
          img1 = img1.resize((8,8))
          img2 = img2.resize((8,8))
          
        #   img = arr.reshape(8, 8)
          img1 = np.array(img1).reshape((-1, 64))
          img2 = np.array(img2).reshape((-1, 64))
          

        #   img = list(arr)
        #   print(img)

          predicted1 = model.predict(img1)
          predicted2 = model.predict(img2)

          if int(predicted1[0]) == int(predicted2[0]):
            return "Predicted Values are Same"
          else:
            return "Predicted Values are Not Same"

        #   return {"y_predicted":int(predicted[0])}
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

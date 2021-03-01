import random
from flask import Flask, request, jsonify,render_template,redirect
from main import CovidClassifier
import os, shutil





app = Flask(__name__,static_url_path="/static")


@app.route("/predict", methods=['POST'])
def predict():
    image = request.files["file"]
    file_name = str(random.randint(0, 100000)) + ".jpg"
    image.save(file_name)
    model = CovidClassifier()
    result= model.predict(file_name)
    print(result)
    os.remove(file_name)
    result = {"result": result}

    running= False

    return jsonify(result)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/action",methods=['POST'])
def action():

    result = " "
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            image = request.files["file"]
            file_name = str(random.randint(0, 100000)) + ".jpg"
            image.save(file_name)
            model = CovidClassifier()
            result = model.predict(file_name)
            print(result)
            os.remove(file_name)
            result = {"result": result}
            result=result['result']



    return render_template('index.html', result=result)


if __name__ == "__main__":

    app.run(debug=False,threaded=True)

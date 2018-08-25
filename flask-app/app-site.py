# https://www.youtube.com/watch?v=MwZwr5Tvyxo
# https://www.tutorialspoint.com/flask/flask_file_uploading.htm
# https://www.youtube.com/watch?v=Y2fMCxLz6wM
# https://www.youtube.com/watch?v=yO_XNCsBIsg
# https://getbootstrap.com/docs/3.3/examples/grid/
# https://www.script-tutorials.com/html5-image-uploader-with-jcrop/
# https://www.w3schools.com/bootstrap/bootstrap_images.asp
from flask import Flask, render_template, request, send_from_directory
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
@app.route("/home")
def upload_file():
    return render_template("upload.html")

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file_2():
    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        dest_path = os.path.join(APP_ROOT, "images", "00target_image.jpg")
        f.save(dest_path)
        image_names = os.listdir('./images')
        return render_template("gallery.html", image_names=image_names)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=True)
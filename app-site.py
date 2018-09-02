from flask import Flask, render_template, request, send_from_directory
import os
from utils.utils import load_embeddings, EmbeddingsAll


app = Flask(__name__)

# load embeddings file previously created
embs_catalogue = load_embeddings('./embeddings.csv')
imgs_catalogue = os.listdir("./images/catalogue")
em = EmbeddingsAll()

@app.route("/")
@app.route("/home")
def upload_file():
    return render_template("upload.html")

@app.after_request
def add_header(response):
    """
    Chache Fix:
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

# transform image uploaded
@app.route('/upload', methods = ['POST'])
def upload_file_2():
    target_img = "./images/target/target_image.jpg"
    recomd_path = "./images/recommend/"
    f = request.files['file']
    f.save(target_img) #saves uploaded image
    top_n = 5 #change to return more than 5 recommendations
    em.top_similar(target_img, recomd_path, embs_catalogue, imgs_catalogue, top_n)

    target_img = target_img.split("/")[-1]  #name of target image
    recomd_names = os.listdir(recomd_path) #name of recommended images
    return render_template("gallery.html", target_image=target_img, image_names=recomd_names)

@app.route('/upload/<filename>')
def send_image(filename):
    if filename == "target_image.jpg":
        return send_from_directory("images/target", filename)
    else:
        return send_from_directory("images/recommend", filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)

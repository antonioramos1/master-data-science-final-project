import os
from flask import Flask, render_template, request, send_from_directory
from utils.utils import load_embeddings, Recommender


app = Flask(__name__)

embs_store = load_embeddings('./embeddings.npy') # load embeddings from previously created file
imgs_store = os.listdir("./images/store")
user_img = "user_img.jpg"
user_img_path = "./images/user/" + user_img
reco_path = "./images/recommend/"
reco = Recommender()

@app.route("/")
@app.route("/home")
def upload_page():
    return render_template("upload.html")

@app.after_request
def fix_cache(response):
    """
    Chache fix: Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/upload', methods = ['POST'])
def upload_image():
    f = request.files['file']
    f.save(user_img_path) #saves uploaded image
    
    top_n = 12
    reco.recommend_user(user_img_path, embs_store, imgs_store, top_n) #will save in the recommend folder the top_n most similar images

    reco_names = os.listdir(reco_path) #name of recommended images
    return render_template("gallery.html", target_image=user_img, image_names=reco_names)

@app.route('/upload/<filename>')
def send_image(filename):
    if filename == user_img:
        return send_from_directory("images/user", filename)
    else:
        return send_from_directory("images/recommend", filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)

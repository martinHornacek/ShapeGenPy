from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Update this to the path of your image sequence folder
IMAGE_FOLDER = '/Users/martin/Documents/ShapeGenPy/results/lena_2024-08-11_13.55.00.837154_sequence'

@app.route('/')
def index():
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))])
    return render_template('viewer.html', image_files=image_files)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, jsonify, send_from_directory, render_template_string
from pathlib import Path
import os

app = Flask(__name__)

@app.route('/')
@app.route('/<tomogram>')
def serve_index(tomogram=None):
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Browser</title>
            <link rel="stylesheet" href="{{ url_for('serve_static', filename='styles.css') }}">
            <script src="https://openseadragon.github.io/openseadragon/openseadragon.min.js"></script>
        </head>
        <body>
            <h1 id="image-title"></h1>
            <div id="openseadragon-viewer"></div>
            <div id="navigation">
                <button id="prev-button">Previous</button>
                <button id="next-button">Next</button>
            </div>
            <script>
                const dataset = "czi";
            </script>
            <script src="{{ url_for('serve_static', dir=dir, filename='script_czi.js') }}"></script>
        </body>
        </html>
    ''', dir=dir)

@app.route('/czi/images/<tomo>/<path:filename>')
def serve_static_images_tomo(tomo, filename):
    image_dir = f'/home/groups/mammoth/processing/CJS/Tomo_processing/ProcessingCJS/tomograms/Unlabeled_tomograms/CZI_TOMOS/{tomo}/aretomo/pngs/'
    return send_from_directory(str(Path(image_dir) ), filename)

@app.route('/static/<filename>')
def serve_static(filename):
    print(filename)
    return send_from_directory('/home/gouauxlab/andecath/findingampa/webapp', filename)

@app.route('/czi/images', methods=['GET'])
def get_images():
    print("Hi")
    image_dir = f'/home/groups/mammoth/processing/CJS/Tomo_processing/ProcessingCJS/tomograms/Unlabeled_tomograms/CZI_TOMOS/'

    az_images = [d.parent.parent.name for d in Path(image_dir).glob('*/aretomo/pngs/') if d.is_dir()]
    print(az_images)
    image_groups = {}
    for tomo_name in az_images:
       
        image_groups[f"{tomo_name}"] = {
            "az": f'/{tomo_name}/{tomo_name}_center.150.png',
            "center": f'/{tomo_name}/{tomo_name}_center.120.png',
            "position": f'/{tomo_name}/{tomo_name}_center.180.png',
            "picks": f'/{tomo_name}/active_zonogram_selected_aunps.png',
        }
    return jsonify(image_groups)

if __name__ == '__main__':
    app.run(debug=True, port=5010, host='0.0.0.0')

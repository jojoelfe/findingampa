from flask import Flask, jsonify, send_from_directory, render_template_string
from pathlib import Path
import os

app = Flask(__name__)

@app.route('/')
def landing_page():
    base_dir = Path('/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/')
    directories = [d.name for d in base_dir.iterdir() if d.is_dir()]
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Landing Page</title>
            <link rel="stylesheet" href="styles.css">
        </head>
        <body>
            <h1>Available Directories</h1>
            <ul>
                {% for dir in directories %}
                    <li><a href="/{{ dir }}/">{{ dir }}</a></li>
                {% endfor %}
            </ul>
        </body>
        </html>
    ''', directories=directories)

@app.route('/<dir>/')
def serve_index(dir):
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Browser</title>
            <link rel="stylesheet" href="{{ url_for('serve_static', dir=dir, filename='styles.css') }}">
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
                const dataset = "{{ dir }}";
            </script>
            <script src="{{ url_for('serve_static', dir=dir, filename='script.js') }}"></script>
        </body>
        </html>
    ''', dir=dir)

@app.route('/<dir>/images/<path:filename>')
def serve_static_images(dir, filename):
    image_dir = f'/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/{dir}/TOP_TOMOS/tomogram_overview_images/'
    print("Mine",image_dir,filename)
    return send_from_directory(image_dir, filename)

@app.route('/<dir>/static/<path:filename>')
def serve_static(dir,filename):
    print("Thine",dir, filename)
    return send_from_directory('.', filename)

@app.route('/<dir>/images', methods=['GET'])
def get_images(dir):
    image_dir = f'/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/{dir}/TOP_TOMOS/tomogram_overview_images/'
    
    az_images = list(Path(image_dir).glob('*_active_zonogram_[0-9].png'))
    image_groups = {}
    for image in az_images:
        tomoname = image.stem[:-len('_active_zonogram_0')]
        az_id = int(image.stem[-1])
        image_groups[f"{tomoname}_az{az_id}"] = {
            "az": f'/{image.name}',
            "center": f'/{tomoname}_center.png.187.png',
            "position": f'/{image.stem}_position.png',
            "picks": f'/{image.stem}_selected_aunps.png',
        }
    return jsonify(image_groups)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

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
@app.route('/<dir>/<tomogram>')
def serve_index(dir, tomogram=None):
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
    return send_from_directory(image_dir, filename)

@app.route('/<dir>/images/<tomo>/<path:filename>')
def serve_static_images_tomo(dir, tomo, filename):
    image_dir = f'/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/{dir}/TOP_TOMOS/'
    return send_from_directory(str(Path(image_dir) / tomo / "best_alignment" / "active_zonograms"), filename)

@app.route('/<dir>/static/<path:filename>')
def serve_static(dir,filename):
    return send_from_directory('.', filename)

good_images = {
    "15F1_tomograms": [
"20230512-EG18pos2_80",
"20231017_EGmilled24-2_60",
"20231017_EGmilled24-2_68",
"20231017_EGmilled24-2_77",
"20231017_HippAu_132",
"20231017_HippAu_141",
"20231017_HippAu_150",
"20231026_HippAu_14",
"20231026_HippAu_26",
"20231026_HippAu_30",
"20240111_WaffleHipp_116",
"20240111_WaffleHipp_127",
"20240111_WaffleHipp_128",
"20240111_WaffleHipp_129",
"20240111_WaffleHipp_132",
"20240111_WaffleHipp_133",
"20240111_WaffleHipp_134",
"20240111_WaffleHipp_135",
"20240111_WaffleHipp_138",
"20240111_WaffleHipp_179",
"20240111_WaffleHipp_183",
"20240111_WaffleHipp_227",
"20240111_WaffleHipp_91",
"20240111_WaffleHipp_96",
"20240209_waffleHip_165",
"20240209_waffleHip_35",
"20240209_waffleHip_78",
"20240307_AMmilled6_217",
"20240307_AMmilled6_252",
"20240523_HippWaffle_168",
"20240523_HippWaffle_197",
"20240523_HippWaffle_71",
"20240523_HippWaffle_91",
"20240624_HippWaffle_83",
"20240624_HippWaffle_86",
"20240901_AMmilled13-1_29",
"20240901_AMmilled13-1_43",
"20241030_AMmilled12-1_15",
"20241030_AMmilled12-2_78",
"20241031_AMmilled13-1_19",
"20241031_AMmilled13-1_21",
"20240111_WaffleHipp_154",
"20240111_WaffleHipp_198",
"20240624_HippWaffle_21",
"20241030_AMmilled12-2_53",
"20241030_AMmilled12-2_62",
"20231017_HippAu_159",
"20240111_WaffleHipp_150",
"20240111_WaffleHipp_69",
"20240624_HippWaffle_18",
"20240624_HippWaffle_22",
"20241030_AMmilled12-2_30",
"20241208_AMmilled20-1_40",
"20240111_AMmilled4-1_140",
"20240209_waffleHip_71",
"20240624_HippWaffle_88",
"20241030_AMmilled12-2_41"
    ]
}

@app.route('/<dir>/images', methods=['GET'])
def get_images(dir):
    image_dir = f'/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/{dir}/TOP_TOMOS/'
    
    az_images = good_images[dir]
    image_groups = {}
    for tomo_name in az_images:
        azs = list((Path(image_dir) / tomo_name / 'best_alignment' / 'active_zonograms').glob('active_zonogram_*.mrc'))
        for az in azs:
            az_id = int(az.stem[-1])
            image_groups[f"{tomo_name}_az{az_id}"] = {
                "az": f'/{tomo_name}/active_zonogram_{az_id}.png',
                "center": f'/{tomo_name}_center.png.187.png',
                "position": f'/{tomo_name}/active_zonogram_{az_id}_position.png',
                "picks": f'/{tomo_name}/active_zonogram_{az_id}_selected_aunps.png',
            }
    return jsonify(image_groups)

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')

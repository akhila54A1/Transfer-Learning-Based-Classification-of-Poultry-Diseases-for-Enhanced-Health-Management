from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Dummy poultry classes
poultry_classes = ['Coccidiosis', 'Newcastle Disease', 'Salmonella', 'Healthy']

def model_predict(image_path):
    # Dummy prediction: always return 'Coccidiosis'
    return 'Coccidiosis', {'Coccidiosis': 0.95, 'Newcastle Disease': 0.03, 'Salmonella': 0.01, 'Healthy': 0.01}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction, probabilities = model_predict(filepath)
        return render_template('predict.html', prediction=prediction, image_path=filepath)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

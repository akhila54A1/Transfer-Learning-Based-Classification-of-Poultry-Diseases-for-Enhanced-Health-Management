from flask import Flask, request, render_template_string, session, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import cv2
import base64
import io

app = Flask(__name__)
app.secret_key = 'my_secret_key'

# Load your trained model
model = joblib.load("decision_tree_model.pkl")
MODEL_ACCURACY = 70.14  # your model's accuracy

# Convert uploaded image to base64 string to display in HTML
def image_to_base64(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((100, 100))
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded, np.array(image)

# Flatten and preprocess the image
def preprocess_image(image_np):
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    features = image_np.flatten().reshape(1, -1)
    return features

LOGIN_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: url('/static/bg.jpg') no-repeat center center fixed;
            background-size: cover;
        }
        .login-box {
            background: rgba(255, 255, 255, 0.9);
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="login-box">
        <h2>Login</h2>
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required><br>
            <input type="password" name="password" placeholder="Password" required><br>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
'''

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🐔 Poultry Disease Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: url('https://static.vecteezy.com/system/resources/thumbnails/050/036/611/small/close-up-chicken-in-poultry-farming-photo.jpg');
            background-size: cover;
            margin: 0; padding: 0;
            display: flex; justify-content: center; align-items: center;
            min-height: 100vh;
        }
        .card {
            background: #ffffff;
            border: 3px solid #4caf50;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            max-width: 500px;
            width: 90%;
            text-align: center;
            position: relative;
        }
        h1 {
            color: #2e7d32;
            margin-bottom: 10px;
        }
        input[type="file"] {
            margin: 15px 0;
        }
        button {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 12px 25px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s;
        }
        button:hover {
            background-color: #388e3c;
        }
        .result, .accuracy {
            margin-top: 20px;
            font-size: 18px;
            color: #0d47a1;
        }
        .image-preview {
            margin-top: 20px;
        }
        .image-preview img {
            max-width: 100%;
            border: 2px solid #ccc;
            border-radius: 8px;
        }
        .history {
            margin-top: 30px;
            text-align: left;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4caf50;
            color: white;
        }
        .logout-btn {
            position: absolute;
            top: 10px;
            right: 10px;
        }
    </style>
</head>
<body>
    <div class="card">
        <form action="/logout" method="get" class="logout-btn">
            <button type="submit">Logout</button>
        </form>
        <h1>🐔 Poultry Disease Detector</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required><br>
            <button type="submit">Predict Disease</button>
        </form>

        {% if prediction %}
            <div class="image-preview">
                <h3>🖼 Uploaded Image:</h3>
                <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="Uploaded Image">
            </div>
            <div class="result">✅ Predicted Disease: <strong>{{ prediction }}</strong></div>
            <div class="accuracy">🎯 Model Accuracy: {{ accuracy }}%</div>
        {% endif %}

        {% if history %}
        <div class="history">
            <h3>🔓 Prediction History</h3>
            <table>
                <tr><th>#</th><th>Disease</th></tr>
                {% for i, item in enumerate(history) %}
                    <tr><td>{{ i+1 }}</td><td>{{ item }}</td></tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] == 'Harshitha' and request.form['password'] == 'HarshithA':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid username or password'
    return render_template_string(LOGIN_PAGE, error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    prediction = None
    image_base64 = None
    probability = None

    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_base64, image_np = image_to_base64(file)
            features = preprocess_image(image_np)
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)
            probability = round(np.max(proba) * 100, 2)
            session['history'].append(prediction)
            session.modified = True

    return render_template_string(
        HTML_PAGE,
        image_base64=image_base64,
        prediction=prediction,
        accuracy=MODEL_ACCURACY,
        probability=probability,
        history=session.get('history', []),
        enumerate=enumerate
    )

if __name__ == '__main__':
    app.run(debug=True)

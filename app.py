import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

MODEL_PATH = 'brain_tumor_model.pkl'
SCALER_PATH = 'scaler.pkl'

def create_and_train_model():
    print("Training model... This may take a few minutes.")
    from sklearn.model_selection import train_test_split
    
    image_directory = 'Dataset/'
    no_tumor_images = os.listdir(image_directory + 'no/')
    yes_tumor_images = os.listdir(image_directory + 'yes/')
    
    dataset = []
    label = []
    INPUT_SIZE = 64
    
    for i, image_name in enumerate(no_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'no/' + image_name)
            if image is not None:
                image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
                dataset.append(image.flatten())
                label.append(0)
    
    for i, image_name in enumerate(yes_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'yes/' + image_name)
            if image is not None:
                image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
                dataset.append(image.flatten())
                label.append(1)
    
    X = np.array(dataset)
    y = np.array(label)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained! Accuracy: {accuracy:.4f}")
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return model, scaler

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Loading existing model...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    else:
        return create_and_train_model()

model, scaler = load_or_train_model()
print('Model ready. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img):
    image = cv2.imread(img)
    if image is None:
        return 0
    image = cv2.resize(image, (64, 64))
    image = image.flatten()
    image_scaled = scaler.transform([image])
    prediction = model.predict(image_scaled)
    return prediction[0]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, redirect, request, url_for # Added url_for for redirects
import mysql.connector, os
import numpy as np # Not directly used in the snippet, but kept if needed elsewhere
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from werkzeug.utils import secure_filename
# import timm # Not used in the provided snippet, removed for clarity if not needed

app = Flask(__name__)

# --- GLOBAL VARIABLE FOR USER SESSION (CAUTION: Not production-ready, use Flask-Login for proper sessions) ---
user_email = None

# --- DATABASE CONNECTION ---
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='Ulcer'
)
mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html') # Assuming this is your landing page

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['username']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['confirm_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Confirm password does not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            # IMPORTANT: For security, never store or compare plain text passwords.
            # Use password hashing (e.g., werkzeug.security.generate_password_hash/check_password_hash)
            query = "SELECT password FROM users WHERE email = %s" # Assuming password column stores plain text now
            values = (email,)
            password_db = retrivequery1(query, values) # Renamed to avoid conflict

            if password == password_db[0][0]: # Direct comparison of plain text passwords
                global user_email # Declare intent to modify global variable
                user_email = email # Set user's email upon successful login
                print(f"User {user_email} logged in.") # For debugging
                return redirect(url_for("home")) # Redirect to home after successful login
            return render_template('login.html', message="Invalid Password!!")
        return render_template('login.html', message="This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html') # This is where users go after logging in


# --- FLASK CONFIGURATION ---
# Define the UPLOAD_FOLDER relative to your Flask app's root
UPLOAD_FOLDER = os.path.join('static', 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------- DEVICE ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- CLASS NAMES ---------------
class_names = ["Normal", "Abnormal"] # Assuming "Normal" means no ulcer, "Abnormal" means ulcer

# --------------- TRANSFORM ---------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------- MODEL DEFINITION ---------------
class CCDGS(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CCDGS, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.channel_attn(y).view(b, c, 1, 1)
        x_channel = x * y
        s = self.spatial_attn(x_channel)
        x_spatial = x_channel * s
        return x_spatial

class DenseNet169_CCDGS(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet169_CCDGS, self).__init__()
        base_model = models.densenet169(pretrained=True)
        self.features = base_model.features
        self.ccdgs = CCDGS(in_channels=1664)
        self.classifier = nn.Linear(base_model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.ccdgs(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --------------- LOAD MODEL ---------------
# Ensure 'best_densenet_ccdgs.pt' is in the same directory as your app.py
try:
    model = DenseNet169_CCDGS(num_classes=2).to(device)
    model.load_state_dict(torch.load("best_densenet_ccdgs.pt", map_location=device))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: best_densenet_ccdgs.pt not found. Please ensure the model file is in the same directory as app.py")
    model = None # Set model to None to prevent subsequent errors

# --------------- PREDICTION FUNCTION ---------------
def predict_image(image_path):
    if model is None:
        # Handle case where model failed to load
        return "Error: Model not loaded.", 0.0, None

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    predicted_class = class_names[pred.item()]
    confidence = probs[0][pred.item()].item()
    return predicted_class, confidence, image_path # image_path is the full server path here

# --------------- FLASK ROUTES ---------------

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    predicted_label = None
    confidence = None
    path_for_html = None # This variable will hold the path suitable for HTML's url_for
    message = None

    if request.method == 'POST':
        if 'image' not in request.files:
            message = 'No image file part in the request.'
        else:
            uploaded_file = request.files['image']
            if uploaded_file.filename == '':
                message = 'No selected file.'
            else:
                if uploaded_file:
                    filename = secure_filename(uploaded_file.filename)
                    full_server_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    uploaded_file.save(full_server_filepath)

                    path_for_html = os.path.join('upload', filename).replace('\\', '/')

                    predicted_label, confidence, _ = predict_image(full_server_filepath)

                    # Save prediction to DB ONLY if a user is logged in
                    global user_email # Access the global user_email
                    if user_email: # Check if user_email is set (user is logged in)
                        query = "INSERT INTO predictions (email, image_path, prediction, confidence) VALUES (%s, %s, %s, %s)"
                        values = (user_email, path_for_html, predicted_label, round(confidence * 100, 2))
                        executionquery(query, values)
                    else:
                        message = "Prediction made, but not saved to your profile. Please log in to save results."

                    print(f"Prediction: {predicted_label}, Confidence: {confidence*100:.2f}%")
                    print(f"Full Server Filepath: {full_server_filepath}")
                    print(f"Path for HTML (url_for): {path_for_html}")

    return render_template('upload.html',
                           path=path_for_html,
                           prediction=predicted_label,
                           confidence=round(confidence * 100, 2) if confidence is not None else None,
                           message=message)

@app.route('/results')
def results():
    global user_email # Access the global user_email

    results_data = [] # Initialize to an empty list

    if user_email: # Check if a user is logged in
        # Fetch predictions ONLY for the logged-in user's email
        query = "SELECT email, image_path, prediction, confidence, uploaded_at FROM predictions WHERE email = %s ORDER BY uploaded_at DESC LIMIT 10"
        values = (user_email,)
        results_data = retrivequery1(query, values) # Use retrivequery1 for parameterized queries
        print(f"Displaying results for user: {user_email}. Found {len(results_data)} predictions.") # Debugging
    else:
        # If no user is logged in, redirect to login page with a message
        print("Attempted to access results without being logged in. Redirecting to login.") # Debugging
        return render_template('login.html', message="Please log in to view your prediction history.")

    return render_template('results.html', results=results_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000) # You can change the port if 5000 is in use
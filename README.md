This project implements a **TabTransformer model** using **Keras + TensorFlow** to detect Distributed Denial of Service (DDoS) attacks from tabular network traffic data. It supports **4-class classification** with stratified data splitting for balanced evaluation.  

Before running the project, you need Python and pip installed.  

**Install Python** (Windows):  
1. Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)  
2. Run the installer.  
3. ✅ Check the box **"Add Python to PATH"** during installation.  

**Install Dependencies:**  
pip install -r requirements.txt

Train the Model:
python ddos_detection.py

Load and Use the Model:

from tensorflow.keras.models import load_model
# Load trained model
model = load_model("ddosAttackDetection.h5")

# Predict on new data
predictions = model.predict(new_data)


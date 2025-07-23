Cotton Leaf Disease Detection Using Artificial Intelligence with Autonomous Alerting System 🌱🤖
This project presents an AI-powered system for early detection and classification of cotton leaf diseases using Convolutional Neural Networks (CNN), integrated with an autonomous robotic vehicle for real-time pesticide application and farmer alerting.

📌 Features
🧠 CNN-based model with 99.997% accuracy on cotton disease classification
📷 Real-time leaf image analysis via webcam or drone input
📡 Automated alert system (SMS-based) for farmers
🤖 Robot with Raspberry Pi + ZigBee module to spray pesticides autonomously
🌿 Classifies 6 common cotton leaf diseases + healthy leaf
🌐 Flask-based web interface for user interaction

🧪 Diseases Detected
Aphids
Armyworm
Bacterial Blight
Powdery Mildew
Target Spot
Fusarium Wilt
Healthy Leaf

🛠️ Technologies Used
Domain	Tools & Libraries
Programming	Python, Arduino C
Deep Learning	TensorFlow, Keras, OpenCV
Frontend	Flask (Python Web Framework)
Hardware	Raspberry Pi, ZigBee, Motor Driver, Sprayer
Dataset	Kaggle - Cotton Leaf Disease Dataset

🚀 Project Architecture
User Uploads Image --> CNN Classifier --> Disease Detection
       ↓                                  ↓
Autonomous Robot Notified         Farmer Notified via SMS
       ↓
Robot Moves to Affected Area
       ↓
Sprays Pesticide Automatically

🧩 How It Works
Image Input: Captured using high-res cameras or drones.
Preprocessing: RGB to Grayscale conversion, noise reduction, thresholding.
CNN Classification: Uses pre-trained CNN model to identify disease.
Farmer Alert: SMS with disease info + remedy is sent.
Automated Action: Robot navigates to location and applies pesticide.

📊 Results
Dataset: 3,557 training images, 84 testing images.
Accuracy: Achieved 99.997% classification accuracy using CNN.
Testing Platform: Raspberry Pi integrated with hardware modules.

🧱 System Requirements
Software
Python 3.x
TensorFlow, Keras, OpenCV, Flask
Arduino IDE (for hardware control)
Hardware
Raspberry Pi (3 or 4)
ZigBee Transceiver
Camera Module (HD)
Motor Driver, Wheels
Pesticide Sprayer Mechanism

🧪 Setup Instructions
# Clone the repository
git clone https://github.com/yourusername/cotton-leaf-disease-ai.git
cd cotton-leaf-disease-ai

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

📱 Future Work
Mobile app integration for real-time monitoring

Enhanced robot navigation using GPS/vision

Expand disease database to other crops

Integration with IoT sensors for climate/environmental analysis

📚 References
Refer to the full research report and publication in the World Conference on Communication & Computing 2023 for detailed insights.
https://ieeexplore.ieee.org/abstract/document/10235089


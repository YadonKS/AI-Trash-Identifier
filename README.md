![image](https://github.com/user-attachments/assets/6bdacde4-1582-40e1-860b-aab79e154627)

# AI-Trash-Identifier
An AI-powered web application that identifies and classifies types of trash from user-uploaded images, designed to support better waste sorting and recycling efforts.

# Overview
This application will use the CNN model, which is built using TensorFlow. It will categorize the images of trash into two categories. The model has been trained on the labeled dataset from Kaggle to be able to recognize common types of waste. A web server connects a model to a HTML/CSS/JavaScript front-end. Users are able to upload an image and get real-time prediction about trash type.

# Features
**Image Classification**: Classifies images of trash into categories, aiding waste management efforts.

**User-Friendly Interface**: Simple web interface built with HTML, CSS, and JavaScript for easy image uploads and results display.

**Real-Time Prediction**: Fast, accurate results powered by a trained CNN model integrated with Flask.

# Installation
1. Clone this repository.
2. Install the required packages:
  pip install -r requirements.txt
3. Start the Flask server:
  python app.py

# Usage
1. Launch the app by navigating to http://localhost:5000.
2. Upload an image of trash through the interface.
3. Receive an immediate prediction of the trash type.
   
# Technologies Used
- **TensorFlow**: For training and running the CNN model.
- **Flask**: For connecting the front end with the model logic.
- **HTML/CSS/JavaScript**: For building the web interface.

# Acknowledgements
Dataset from Kaggle: https://www.kaggle.com/datasets/techsash/waste-classification-data?resource=download

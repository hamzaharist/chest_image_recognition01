# Chest Image Recognition for COVID, Pneumonia, or Normal Diagnosis
This Streamlit app utilizes a trained machine learning model to classify chest X-ray images into three categories: Viral Pneumonia, COVID-19, or Normal. The underlying model is loaded from a JSON file containing the architecture and a separate weight file.

Instructions
Follow these steps to use the Chest Image Recognition app:

<h3>Clone the Repository: </h3>

bash
Copy code
'''git clone https://github.com/your-username/your-repo.git
cd your-repo
Install Dependencies:
Make sure you have the required dependencies installed. You can install them using the following command:

bash
Copy code
'''pip install -r requirements.txt
Run the Streamlit App:
Execute the following command to run the Streamlit app:

bash
Copy code
streamlit run main.py
This will launch a local development server, and you can access the app in your web browser.

##Upload Chest X-ray Images:
Once the app is running, you can upload chest X-ray images in JPG, JPEG, or PNG format.

##View Diagnosis:
The app will display the uploaded image and provide a diagnosis, categorizing it as Viral Pneumonia, COVID-19, or Normal.

##Model Architecture
The underlying machine learning model is stored in two files: model.json (containing the architecture) and model.h5 (containing the weights). The model is loaded using TensorFlow's Keras API.

##Background Image
The background of the app is set using the set_background function, allowing for a visually appealing and themed user interface.

Feel free to customize and extend this app based on your specific requirements. If you encounter any issues or have suggestions for improvements, please let us know!

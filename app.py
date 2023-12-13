import streamlit as st
import base64
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define diagnosis mapping globally
diagnosis_mapping = {0: 'Viral Pneumonia', 1: 'Covid', 2: 'Normal'}

# Function to load the model
def load_model():
    # Load model architecture from JSON file
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    # Close the JSON file
    model = model_from_json(loaded_model_json)

    # Load weights into the new model
    model.load_weights('model.h5')

    return model

# Function to make a diagnosis
def diagnosis(file, model, IMM_SIZE):
    # Load and preprocess the image
    img = image.load_img(file, target_size=(IMM_SIZE, IMM_SIZE), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]

    # Predict the diagnosis and confidence score
    predicted_probabilities = model.predict(img_array)
    predicted_class = np.argmax(predicted_probabilities, axis=-1)[0]

    # Map the predicted class to the diagnosis
    predicted_diagnosis = diagnosis_mapping[predicted_class]

    return predicted_diagnosis

# Function to set the background
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Function to compute performance metrics
def compute_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return accuracy, precision, recall, f1

# Main Streamlit app
def main():
    set_background('bg5.png')
    
    st.title("Chest X-Ray Predictor")
    st.markdown("""
    Welcome to the Chest X-Ray Predictor! Upload a chest X-ray image, and we will predict its diagnosis.
    
    The diagnosis will be one of the following categories:
    - Viral Pneumonia
    - Covid
    - Normal

    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        # Load the model
        model = load_model()

        # Specify the image size
        IMM_SIZE = 224

        try:
            # Get diagnosis and confidence score
            result = diagnosis(uploaded_file, model, IMM_SIZE)

            # Display the result and confidence score
            st.write("## Diagnosis: {}".format(result))

            # Add true labels from your dataset
            true_labels = [1, 0, 2, ...]  # Replace with actual ground truth labels

            # Map predicted diagnosis to numerical values
            predicted_class = list(diagnosis_mapping.keys())[list(diagnosis_mapping.values()).index(result)]

            # Compute performance metrics
            accuracy, precision, recall, f1 = compute_metrics(true_labels, [predicted_class])

            # Display performance metrics
            st.write("## Performance Metrics")
            st.write("### Accuracy: {:.2%}".format(accuracy))
            st.write("### Precision: {:.2%}".format(precision))
            st.write("### Recall: {:.2%}".format(recall))
            st.write("### F1 Score: {:.2%}".format(f1))
            
        except Exception as e:
            st.error(f"Error during diagnosis: {e}")
            print("Error during diagnosis:", e)

# Run the app
if __name__ == "__main__":
    main()

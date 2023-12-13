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

            # Ensure that the lengths of true_labels and [predicted_class] are the same
            assert len(true_labels) == len([predicted_class]), "Length mismatch between true_labels and predicted_class"

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

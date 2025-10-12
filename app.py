import streamlit as st
import cv2  
from PIL import Image, ImageEnhance
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import tensorflow as tf
import base64


def main():
    st.set_page_config(
         page_title="Alzheimer's detection",
         page_icon="🧠",
         layout="centered")

    # Try to load video if it exists, otherwise use a simple background
    try:
        video_file = 'tech.mp4'
        video_bytes = open(video_file, 'rb').read()
        encoded_video = base64.b64encode(video_bytes).decode()

        video_html = f"""
        <style>
        #myVideo {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        }}

        .content {{
        position: fixed;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        color: #f1f1f1;
        width: 100%;
        padding: 20px;
        }}
        </style>
        <video autoplay muted loop id="myVideo">
        <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        Your browser does not support HTML5 video.
        </video>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback to a simple dark background
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
        }
        </style>
        """, unsafe_allow_html=True)

    
    tasks = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Task', tasks)

    # Try to load default image, but don't fail if it's not found
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        img_pth = os.path.join(root_dir, "3.jpg")
        img = Image.open(img_pth)
    except FileNotFoundError:
        img = None

    if choice == 'Detection':
        st.markdown("<h1 style='text-align: center; color: white;'><i>Alzheimer's Detection Tool🧠</i></h1>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center; color: white;'><i>A stitch in time saves a nine!</i></h6>", unsafe_allow_html=True)
        tasks = ["Alzheimer's"]
        if tasks[0] == "Alzheimer's":
            # Load the saved model
            try:
                loaded_model = tf.keras.models.load_model("model.h5",compile=False)
            except FileNotFoundError:
                st.error("Model file not found. Please ensure model.h5 exists in the workspace.")
                return

            def predict_image_class(img_path):
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                predictions = loaded_model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]

                return predicted_class

            uploaded_file = st.file_uploader('Upload an MRI image...', type=['jpg', 'png', 'jpeg'])

            if uploaded_file is not None:
                try:
                    predicted_class = predict_image_class(uploaded_file)
                    num_classes = loaded_model.layers[-1].output_shape[1]
                    classes = [str(i) for i in range(num_classes)]
                    class_names = ["MILD DEMENTED", "MODERATE DEMENTED", "NON DEMENTED", "VERY MILD DEMENTED"]
                    st.markdown(f"<div style='text-align: center; font-weight: thin;font-size: 30px;'>Predicted Alzheimer's stage is : <b>{class_names[predicted_class]}</b></div>", unsafe_allow_html=True)
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(uploaded_file, use_column_width=False, width=300)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            elif img is not None:
                st.image(img)

    elif choice == 'About':
       
       
        st.image("3.jpg")
        st.header("📊 Early Stage Alzheimer's prediction ")
        st.subheader("🧠What is Alzheimer's?")
        st.write("Alzheimer's disease is a progressive neurodegenerative disorder that primarily affects the brain. It is the most common form of dementia, characterized by the gradual loss of cognitive functions, including memory, thinking, and language. The disease is caused by the buildup of abnormal proteins, such as amyloid-beta and tau, which form plaques and tangles in the brain, leading to the death of nerve cells and the deterioration of brain tissue. As the disease progresses, individuals experience difficulty with daily activities, mood changes, and ultimately, a severe decline in their overall cognitive and functional abilities. Currently, there is no cure for Alzheimer's, but researchers are working to develop new treatments and therapies to slow the progression of the disease and improve the quality of life for those affected.")

        st.subheader("🖥️Deep Learning and Alzheimer's")
        st.write("Deep learning, a branch of artificial intelligence, has shown promising results in the early diagnosis of Alzheimer's disease. Researchers have developed deep learning algorithms that can analyze medical imaging data, such as MRI or PET scans, to detect subtle changes in the brain structure and function that are associated with the onset of Alzheimer's. These algorithms can identify patterns and features in the data that are not easily discernible to the human eye, allowing for earlier detection of the disease before significant cognitive decline occurs. By leveraging the power of deep learning, clinicians can potentially diagnose Alzheimer's at earlier stages, enabling timely interventions and potentially improving patient outcomes.")

        st.subheader("🗄️ Datasets")

        aptos_image = Image.open("istock.jpg")
          
        dr1 = Image.open("11.png")
        dr2 = Image.open("22.png")
        dr3 = Image.open("33.png")

       
        st.image(aptos_image)
        st.write("""The Kaggle Alzheimer's Dataset is a publicly available dataset that has been widely used in research on the early detection and diagnosis of Alzheimer's disease. The dataset consists of brain MRI scans from several hundred participants, along with associated clinical data, such as the participant's age, gender, and cognitive test scores. The MRI scans are labeled according to the participant's clinical diagnosis, including healthy controls, as well as individuals with mild cognitive impairment and Alzheimer's disease. 
Researchers and data scientists have leveraged this dataset to develop and test various machine learning and deep learning algorithms for the automated analysis of brain MRI scans. The goal is to identify specific patterns or features in the brain images that can distinguish between healthy individuals and those with Alzheimer's or other related cognitive disorders. By training and validating models on this dataset, researchers aim to improve the accuracy and efficiency of Alzheimer's diagnosis, which could lead to earlier interventions and better patient outcomes.
The Kaggle Alzheimer's Dataset has been instrumental in advancing the field of Alzheimer's research, as it provides a standardized and well-curated resource for the scientific community to collaborate, share ideas, and push the boundaries of what's possible in the early detection and diagnosis of this debilitating disease.
"""
        )
        st.subheader("🎚️ Alzheimer's Stages")

        data_col1, data_col2, data_col3, data_col4 = st.columns([1, 1, 1, 1])

        with data_col1:
            st.markdown('<div style="text-align:center">Healthy brain</div>', unsafe_allow_html=True)
            st.image(dr1)

        with data_col2:
            st.markdown('<div style="text-align:center">Moderate</div>', unsafe_allow_html=True)
            st.image(dr2)

        with data_col3:
            st.markdown('<div style="text-align:center">Severe</div>', unsafe_allow_html=True)
            st.image(dr3)


        st.subheader("📈 Data Preprocessing")
        st.image('ss2.png')
        st.write("""
        In the application of deep learning for Alzheimer's disease diagnosis using brain MRI images, image preprocessing is a crucial step to ensure the quality and consistency of the data fed into the neural networks.
        One common preprocessing technique is image normalization, which scales the pixel intensities to a common range, typically between 0 and 1 or -1 and 1. This helps to standardize the input data and reduce the impact of variations in image acquisition parameters or scanner differences.
        Skull stripping is another important preprocessing step, where the non-brain tissues such as the skull, scalp, and dura mater are removed from the MRI images. This can be done using specialized algorithms that segment the brain tissue from the surrounding structures, allowing the deep learning models to focus solely on the relevant brain regions.
        Image registration is also commonly applied, which aligns the MRI scans to a common reference space, such as a standard brain template. This ensures that corresponding anatomical structures are spatially aligned across different patient scans, enabling more accurate comparison and feature extraction.
        Additionally, data augmentation techniques, such as random rotations, flips, or intensity variations, can be applied to the preprocessed MRI images. This can help to increase the size and diversity of the training dataset, improving the generalization capabilities of the deep learning models and reducing the risk of overfitting.
        By carefully applying these preprocessing steps, the input MRI data can be optimized for deep learning analysis, enhancing the performance and robustness of the models in detecting and classifying Alzheimer's disease patterns from brain imaging data.
        """
        )

        

        st.subheader("👨‍💻 Model Training")
        st.write("""To train a deep learning model using EfficientNetB3 for predicting Alzheimer's disease from MRI scans, the following steps can be followed:
1. **Data Preprocessing**:
   - Normalize the MRI scans to a common range (e.g., 0-1 or -1 to 1) to standardize the input data.
   - Perform skull stripping to remove non-brain tissues and focus the model on the relevant brain regions.
   - Apply image registration to align the MRI scans to a common reference space for accurate comparison.
   - Optionally, use data augmentation techniques like random rotations, flips, or intensity variations to increase the size and diversity of the training dataset.
2. **Dataset Preparation**:
   - Split the preprocessed MRI scans into training, validation, and test sets.
   - Label the scans based on the patient's clinical diagnosis (e.g., healthy, mild cognitive impairment, Alzheimer's).
3. **Model Architecture**:
   - Use the EfficientNetB3 model as the base architecture.
   - Modify the final layers of the model to fit the specific task of Alzheimer's disease classification.
4. **Transfer Learning**:
   - Initialize the model with weights pre-trained on a large-scale dataset like ImageNet.
   - Fine-tune the model on the Alzheimer's MRI dataset to adapt the learned features to the specific task.
5. **Training and Optimization**:
   - Train the model on the preprocessed MRI scans, minimizing the classification loss and maximizing the accuracy.
   - Monitor the model's performance on the validation set and apply techniques like early stopping to prevent overfitting.
6. **Evaluation**:
   - Assess the trained model's performance on the held-out test set.
   - Analyze the model's ability to accurately classify different stages of Alzheimer's disease.
By following these steps, you can leverage the power of the EfficientNetB3 model to develop a deep learning-based system for early and accurate Alzheimer's disease prediction using MRI scans. The model's high accuracy and efficient inference make it a valuable tool in the field of Alzheimer's research and clinical practice.
""")
        
        st.image('output.png')
        st.image('ss.png')
    with st.sidebar:
        st.title("Early stage Alzeimer's prediction tool")


if __name__ == '__main__':
    main()
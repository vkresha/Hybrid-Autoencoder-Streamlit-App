import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf  # Ensure TensorFlow is imported
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
from keras.utils import custom_object_scope
from tensorflow.keras.metrics import MeanAbsoluteError



class HybridAutoEncoder(Model):
    def __init__(self, input_dim, latent_dim, num_classes=5):
        super(HybridAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.Conv1D(64, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
            layers.LSTM(latent_dim, activation='tanh', return_sequences=False)
        ])

        from math import ceil
        self.reduced_time_dim = ceil(input_dim / 4)

        self.decoder = tf.keras.Sequential([
            layers.RepeatVector(self.reduced_time_dim),
            layers.LSTM(latent_dim, activation='tanh', return_sequences=True),
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.UpSampling1D(2),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, strides=1, activation='relu', padding="same"),
            layers.UpSampling1D(2),
            layers.BatchNormalization(),
            layers.Conv1D(1, 3, strides=1, activation='sigmoid', padding="same")
        ])

        # Classification head
        self.classifier = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        dummy_input = tf.keras.Input(shape=(input_dim, 1))
        self(dummy_input)

    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        classification_output = self.classifier(encoded)

        return {"reconstruction": decoded, "classification": classification_output}

    def get_config(self):
        config = super(HybridAutoEncoder, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "num_classes": self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            input_dim=config.get("input_dim", 280),  # Default to 280 if not provided
            latent_dim=config.get("latent_dim", 32),  # Default to 32 if not provided
            num_classes=config.get("num_classes", 5)  # Default to 5 if not provided
        )


def load_h5_model(model_path):
    with custom_object_scope({
        'HybridAutoEncoder': HybridAutoEncoder,
        'mae': MeanAbsoluteError()  # Explicitly register the MAE metric
    }):
        return load_model(model_path)

def preprocess_signal(csv_file):
    data = pd.read_csv(csv_file, header=None).values
    if data.shape[1] != 280:
        st.error("The uploaded CSV file must have exactly 280 columns.")
        return None
    return data.reshape(1, 280, 1).astype(np.float32)

def plot_signal(signal, title="1D Signal Visualization"):
    plt.figure(figsize=(20, 4))
    plt.gca().set_facecolor("#ffffff")  
    plt.plot(signal, label='Signal', color="#1E88E5", linewidth=2)  
    plt.title(title, fontsize=14, color="#ffffff")  
    plt.xlabel("Time", fontsize=12, color="#ffffff")
    plt.ylabel("Amplitude", fontsize=12, color="#ffffff")
    plt.grid(color="#cccccc", linestyle='--', linewidth=0.5)  
    plt.legend(facecolor="#333333", edgecolor="#ffffff", fontsize=10) 
    
    st.pyplot(plt)


def add_custom_css():
    st.markdown("""
        <style>
        /* App Background Gradient */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #31393C, #2176FF) !important;  /* Dark to Blue Crayola gradient */
            color: #ffffff;  /* White text */
            text-align : center;
        }

        /* Titles and Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important; /* Sun Glow */
            text-align : center;
        }

        /* Subheadings or Accents */
        .stMarkdown p {
            color: #ffffff; /* Celestial Blue for subheadings or accents */
        }

        /* File Uploader Styling */
        .stFileUploader {
            background-color: #33A1FD !important; /* Carrot Orange */
            color: #ffffff !important;
            border-radius: 8px;
            padding: 10px;
        }

        /* Buttons */
        button {
            background-color: #2176FF !important; /* Blue Crayola */
            color: #ffffff !important;
        }

        /* Progress Bar Styling */
        .stProgress > div > div {
            background-color: #33A1FD !important; /* Celestial Blue */
        }
        </style>
        """, unsafe_allow_html=True)


CLASS_INFO = {
    0: {"label": "Normal beat", "description": "Your heart has shown a fusion beat, which is a combination of a normal heartbeat and an abnormal one. This can occur when both normal and irregular signals mix together. While this can sometimes happen naturally, it’s important to check with your healthcare provider to see if it might be a sign of any underlying issues with your heart rhythm.", "color": "#4CAF50"},  # Green
    1: {"label": "Supraventricular ectopic beat", "description": "Your heart rhythm is normal, meaning the electrical signals in your heart are functioning as they should. This is a positive result and typically indicates that your heart is healthy and working well.", "color": "#FFC107"},  # Yellow
    2: {"label": "Ventricular ectopic beat", "description": "Your heart has shown signs of supraventricular beats, which originate from the upper chambers of the heart, like the atria. This could be due to conditions such as atrial fibrillation or other arrhythmias that affect the heart’s upper chambers. While not always dangerous, these types of irregularities may need further investigation to ensure your heart stays healthy.", "color": "#FF9800"},  # Orange
    3: {"label": "Fusion beat", "description": "Your heart rhythm has shown ventricular beats, which come from the lower chambers (ventricles) of your heart. Some ventricular beats can be harmless, but they could also be linked to more serious conditions, such as ventricular tachycardia, which can affect the heart’s ability to pump blood properly. It’s important to follow up with a healthcare provider to assess the situation and decide on the best course of action.", "color": "#FF5722"},  # Red
    4: {"label": "Unclassified beat", "description": "Your heart’s rhythm was unclassified, meaning the signal was unclear or difficult to categorize. This could be due to interference, a weak signal, or other factors. It’s a good idea to speak with your healthcare provider, who may recommend further tests or monitoring to get a clearer picture of your heart’s health.", "color": "#F44336"},  # Dark Red
}

# Function to Display Severity Alert
def severity_alert(predicted_class):
    class_info = CLASS_INFO.get(predicted_class, {"label": "Unknown", "description": "No description available.", "color": "#9E9E9E"})
    st.markdown(f"""
        <div style="
            background-color: {class_info['color']};
            border-radius: 5px;
            padding: 10px;
            color: white;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            ">
            Predicted Class: {class_info['label']}
        </div>
    """, unsafe_allow_html=True)


# Function to Display Class Description
def display_class_description(predicted_class):
    class_info = CLASS_INFO.get(predicted_class, {"label": "Unknown", "description": "No description available.", "color": "#9E9E9E"})
    st.markdown(f"""
        <div style="
            background-color: #f5f5f5;
            border: 1px solid {class_info['color']};
            border-radius: 5px;
            padding: 15px;
            margin-top: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            color: #333333;
            font-size: 16px;
            line-height: 1.5;
            ">
            <strong>{class_info['label']}:</strong> {class_info['description']}
        </div>
    """, unsafe_allow_html=True)
    
# Streamlit app
def main():
    add_custom_css()
    st.title("Abnormal Heart Rate Detection")
    st.markdown("A hybrid classfication model to detect heart rate based on ECG signals")

    # Load model
    model_path = 'hybrid_autoencoder_model.h5'  
    model = load_h5_model(model_path)

    uploaded_file = st.file_uploader("Upload a CSV file (280 columns for the signal)", type=["csv"])

    if uploaded_file is not None:
        signal_data = preprocess_signal(uploaded_file)
        if signal_data is not None:
            st.markdown("### Uploaded Signal:")
            plot_signal(signal_data[0, :, 0])

            with st.spinner("🔍 Analyzing the signal..."):
                prediction = model.predict(signal_data)
                reconstruction = prediction["reconstruction"]
                classification = prediction["classification"]
                predicted_class = np.argmax(classification, axis=1)[0]
                confidence = classification[0][predicted_class]

                # Display results
                # st.markdown(f"### Prediction: **Class {predicted_class}**")
                st.markdown(f"Confidence: **{confidence:.2%}**")

                # st.markdown("### Reconstructed Signal:")
                # plot_signal(reconstruction[0, :, 0])
            
             # Display Severity Alert
            severity_alert(predicted_class)

            # Display Class Description
            display_class_description(predicted_class)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Abnormal Heart Rate Detection",
        page_icon="💓",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    main()

# Arrhythmia Detection Streamlit App
Hybrid Classification Model to detect Abnormal Heart Rate

## What is this code?

It's a neural network model that performs:

Classification: Identifies the class of the input data (e.g., one of the five categories ['F', 'N', 'S', 'V', 'Q']). (Uses CNN + LSTM)

## What are the main parts of the model?

Encoder: Compresses the input data into a smaller, meaningful "latent space" representation.
Classifier: Uses the same latent space to determine which class the input belongs to (one of F, N, S, V, Q).

## Explanation of the Model Architecture
### Encoder:
Takes input data and extracts key features using layers like:
Convolutional layers: Identify patterns in the data. LSTM (Long Short-Term Memory): Useful for sequential data (like time series or signals).
The result is a smaller "latent representation" that summarizes the input.

## Classifier:
Takes the latent representation and predicts the class of the input signal (e.g., F, N, S, V, Q).

## What Does the Model Do?
### Input:
A single signal (a heartbeat pattern of length 280).

### Processing:
The signal is compressed into a smaller representation (via the encoder).
The latent space is used to Predict its class (classifier).

### Output:
Class prediction: One of the five classes (F, N, S, V, Q).

Screenshots of Prototype:
<img width="832" alt="image" src="https://github.com/user-attachments/assets/244a7e00-4388-4be2-b8a5-5fa709bac405" />

Streamlit Demo Link : https://drive.google.com/file/d/1Ydj2XPdlfipkFt4AVDZb5MeJ2pB3zYvJ/view?usp=drive_link

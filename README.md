# ANN Customer Churn Prediction

## Project Overview

This project uses an Artificial Neural Network (ANN) to predict customer churn based on customer-related information.

Customer churn refers to customers leaving a company or discontinuing its services. The objective of this project is to use machine learning and deep learning techniques to identify customers who are likely to churn.

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit (if used by the application)

## Project Structure

```text
ANN_project/
│
├── app.py
├── Churn_Modelling.csv
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

- Customer churn prediction using ANN.
- Data preprocessing and feature scaling.
- Encoding categorical features.
- Trained deep learning model.
- Prediction using saved model and preprocessing objects.
- Application interface for making predictions.

## Model and Preprocessing

The project uses a trained Artificial Neural Network model.

The following saved files are used:

- `model.h5` – Trained ANN model.
- `scaler.pkl` – StandardScaler object.
- `label_encoder_gender.pkl` – Gender label encoder.
- `onehot_encoder_geo.pkl` – Geography one-hot encoder.

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ANN-Churn-Prediction.git
```

2. Navigate to the project directory:

```bash
cd ANN-Churn-Prediction
```

3. Create and activate a virtual environment (optional):

```bash
python -m venv .venv
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Run the application using the command appropriate for your framework.

For a standard Python application:

```bash
python app.py
```

For a Streamlit application:

```bash
streamlit run app.py
```

## Dataset

The project uses the `Churn_Modelling.csv` dataset containing customer information and a churn indicator.

## Future Improvements

- Improve prediction accuracy.
- Experiment with different ANN architectures.
- Add model evaluation visualizations.
- Improve the user interface.
- Explore additional customer behavior features.

## Author

Developed as a Deep Learning project using Artificial Neural Networks.


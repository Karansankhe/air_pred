import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Use st.cache_data to cache data loading
@st.cache_data
def load_data():
    # Generate a mock dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'Type': np.random.choice(['L', 'M', 'H'], 1000),
        'Air temperature [K]': np.random.uniform(290, 350, 1000),
        'Process temperature [K]': np.random.uniform(300, 400, 1000),
        'Rotational speed [rpm]': np.random.uniform(1000, 3000, 1000),
        'Torque [Nm]': np.random.uniform(10, 100, 1000),
        'Tool wear [min]': np.random.uniform(0, 250, 1000),
        'Machine failure': np.random.choice([0, 1], 1000)
    })
    
    df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})  # Encoding categorical 'Type'
    return df

# Use st.cache_resource to cache the model training process
@st.cache_resource
def train_model(data):
    X = data[['Type', 'Air temperature [K]', 'Process temperature [K]',
              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    y = data['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Streamlit app
st.title("Machine Failure Prediction")

# Load data
df = load_data()
st.write("Sample data:")
st.write(df.head())

# Train model
model, accuracy = train_model(df)
st.write(f"Model Accuracy: {accuracy:.2f}")

# User input
st.header("Input parameters for prediction")
type_input = st.selectbox("Type (L: 0, M: 1, H: 2)", [0, 1, 2])
air_temp = st.slider("Air temperature [K]", min_value=290, max_value=350, value=300)
process_temp = st.slider("Process temperature [K]", min_value=300, max_value=400, value=310)
rot_speed = st.slider("Rotational speed [rpm]", min_value=1000, max_value=3000, value=2000)
torque = st.slider("Torque [Nm]", min_value=10, max_value=100, value=50)
tool_wear = st.slider("Tool wear [min]", min_value=0, max_value=250, value=100)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[type_input, air_temp, process_temp, rot_speed, torque, tool_wear]])
    prediction = model.predict(input_data)

    st.subheader("Prediction result")
    if prediction[0] == 0:
        st.write("Prediction: No Machine Failure")
    else:
        st.write("Prediction: Machine Failure")

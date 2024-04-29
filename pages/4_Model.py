import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def make_predictions(data):
    with st.spinner("Predicting...."):
        loaded_model = load_model('model1.h5')
        st.success('Model loaded successfully')
        scaler = StandardScaler()
        test_x = scaler.fit_transform(data)
        predictions = loaded_model.predict(test_x)
    st.success('Predictions made successfully')
    return predictions

def main():
    st.set_page_config(page_title='Model', page_icon=':bar_chart:', layout='wide')
    st.header('Model')
    st.divider()
    csv_file = st.file_uploader('Upload CSV file', type=['csv'])
    first_column = st.checkbox('Ignore First Column')
    button = st.button('Predict')

    if button:
        if csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                tmp_csv.write(csv_file.getvalue())
                csv_path = tmp_csv.name
            if first_column:
                data = pd.read_csv(csv_path)
                st.success('Data loaded successfully')
                data = data.iloc[:, 1:]
                predictions = make_predictions(data)
                data['Predicted_Adaptabilty'] = predictions
                st.write("Here is your final DataFrame")
                data
                st.stop()
            data = pd.read_csv(csv_path)
            st.success('Data loaded successfully')
            predictions = make_predictions(data)
            data['Predicted_Adaptabilty'] = predictions
            st.write("Here is your final DataFrame")
            data
if __name__ == '__main__':
    main()
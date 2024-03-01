import pickle
import numpy as np
import streamlit as st
import sklearn
from sklearn.linear_model import LogisticRegression

def main():
    path = "C:\\Users\\abdul\\OneDrive\\Desktop\\minproject\\classifier.pkl"
    diabetes_model = pickle.load(open(path, 'rb'))
    
    # Page title
    st.title('Diabetes Prediction using ML')
    
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        hypertension = st.text_input('Hypertension (0/1)')
    with col3:
        heart_disease = st.text_input('Heart Disease (0/1)')
        
    with col1:
        bmi = st.text_input('BMI value')
    with col2:
        hb_alc = st.text_input('HbAlc Level')
    with col3:
        glucose = st.text_input('Glucose value')

    # Code for Prediction
    diab_diagnosis=''
    diab_diagnosis = diabetes_model.predict([[age, hypertension, heart_disease, bmi, hb_alc, glucose]])
    
    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        input_data = [int(age), int(hypertension), int(heart_disease), float(bmi), float(hb_alc), int(glucose)]
        data = np.asarray(input_data)
        data_reshaped = data.reshape(1, -1)
        
        diab_prediction = diabetes_model.predict(data_reshaped)
        diab_percentage = diabetes_model.predict_proba(data_reshaped)

        prob = np.max(diab_percentage, axis=1)
        max_prob = np.round(prob, 3)
        
        if diab_prediction[0] == 1:
            diab_diagnosis = f'The person is diabetic. Estimated risk: {float(max_prob) * 100}'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Call the main function to run the app
if __name__ == "__main__":
    main()
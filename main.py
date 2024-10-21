import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import plotly.graph_objs as go
import utils  # Importing the functions from utils.py

# Set up OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Function to load models
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load models
xgboost_model = load_model('xgb_model.pkl')
native_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
knn_model = load_model('knn_model.pkl')

# Function to prepare input data for prediction
def prepare_input(credit_score, location, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCreditCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Georgraphy_France': 1 if location == "France" else 0,
        'Georgraphy_Germany': 1 if location == "Germany" else 0,
        'Georgraphy_Spain': 1 if location == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

# Function to make predictions
def make_prediction(input_df, input_dict):
    input_df = input_df.fillna(0)
    input_df = input_df.astype(float)

    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1]
    }

    avg_probability = np.mean(list(probabilities.values()))
    return avg_probability, probabilities

# Function to explain prediction
def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

    Here is the customer's information:
    {input_dict}

    - If the customer has over a 40% risk of churning, generate a 3-sentence explanation of why they are at risk of churning.

    - If the customer has less than a 40% risk of churning, generate a 3-sentence explanation of why they might not be at risk of churning.

    Don't mention the probability of churning or the customers who churned or the machine learning model, or say anything like "Based on the machine learning model and the top 10 most important features." Just explain the prediction. Also, don't include any calculations—just give the explanation simple and clear.
    """

    raw_response = client.chat.completions.create(
        model="LLama-3.2-3b-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return raw_response.choices[0].message.content

# Function to generate email
def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank. You are responsible for
ensuring customers stay with the bank and are incentivized with various offers.

You noticed a customer named {surname} has a {round(probability *
100, 1)}% probability of churning.

Here is the customer's information:
{input_dict}

Here is some explanation as to why the customer might be at risk of churning:
{explanation}

Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

Make sure to list out a set of incentives to stay based on their information,
in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer. 
    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return raw_response.choices[0].message.content

# Streamlit app title
st.title("🧮 Customer Churn Prediction")

# Load dataset
df = pd.read_csv("churn.csv")

# Create list of customers for dropdown
customers = [f"{row['CustomerId']} - {row['Surname']} " for _, row in df.iterrows()]

# Dropdown for customer selection
selected_customer_option = st.selectbox("Select a customer", customers, key='customer_selectbox')

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    col1, col2 = st.columns(2)

    # First column for input fields
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer['CreditScore']),
            key='credit_score'
        )
        location = st.selectbox(
            "Location", 
            ["Spain", "France", "Germany"], 
            index=["Spain", "France", "Germany"].index(selected_customer["Geography"]),
            key='location_selectbox'
        )
        gender = st.radio(
            "Gender", 
            ["Male", "Female"], 
            index=0 if selected_customer['Gender'] == "Male" else 1,
            key='gender_radio'
        )
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer['Age']),
            key='age_input'
        )
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer['Tenure']),
            key='tenure_input'
        )

    # Second column for input fields
    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0, 
            value=float(selected_customer['Balance']),
            key='balance_input'
        )
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer['NumOfProducts']),
            key='products_input'
        )
        has_credit_card = st.checkbox(
            "Has Credit Card",
            value=bool(selected_customer['HasCrCard']),
            key='credit_card_checkbox'
        )
        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']),
            key='active_member_checkbox'
        )
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']),
            key='salary_input'
        )

    # Add a button to trigger the prediction and results display
    if st.button('Show Churn Prediction'):

        # Prepare input data for prediction
        input_df, input_dict = prepare_input(
            credit_score, location, gender, age, tenure, balance, num_products, 
            has_credit_card, is_active_member, estimated_salary
        )

        # Make predictions and display results
        avg_probability, probabilities = make_prediction(input_df, input_dict)

        # Show Gauge and Model Comparison side by side
        st.subheader("📊 Churn Probability and Model Comparison")
        gauge_col, model_col = st.columns(2)

        with gauge_col:
            fig = utils.create_gauge_chart(avg_probability)
            st.plotly_chart(fig, use_container_width=True)

        with model_col:
            fig_probs = utils.create_model_probability_chart(probabilities)
            st.plotly_chart(fig_probs, use_container_width=True)

        # Generate and display explanation
        st.subheader("💡 Explanation of Prediction")
        explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
        st.markdown(explanation)

        # Generate and display email
        st.subheader("📧 Personalized Email")
        email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
        st.markdown(email)

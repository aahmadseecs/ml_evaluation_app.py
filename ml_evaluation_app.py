import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error,
    r2_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: App Layout
st.title("Machine Learning Evaluation App")
st.sidebar.header("Upload Dataset")

# Upload Dataset
uploaded_file = st.sidebar.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Step 2: Load Data
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("### Dataset Preview")
    st.write(data.head())

    # Step 3: Data Preprocessing
    st.sidebar.header("Data Preprocessing")
    target_variable = st.sidebar.selectbox("Select Target Variable", data.columns)
    features = st.sidebar.multiselect("Select Features", [col for col in data.columns if col != target_variable])

    if target_variable and features:
        X = data[features]
        y = data[target_variable]
        
        # Handle missing values
        if st.sidebar.checkbox("Fill Missing Values"):
            X.fillna(X.mean(), inplace=True)
            st.sidebar.write("Missing values filled with column mean.")

        # Train-test split
        test_size = st.sidebar.slider("Test Size (Fraction)", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.write("Data split into training and test sets.")

        # Step 4: Model Selection
        st.sidebar.header("Model Selection")
        model_type = st.sidebar.radio("Select Model Type", ["Classification", "Regression"])

        if model_type == "Classification":
            model = st.sidebar.selectbox("Select Classification Model", [
                "Logistic Regression", "Decision Tree", "Random Forest"
            ])

            if model == "Logistic Regression":
                classifier = LogisticRegression()
            elif model == "Decision Tree":
                classifier = DecisionTreeClassifier()
            elif model == "Random Forest":
                n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
                classifier = RandomForestClassifier(n_estimators=n_estimators)

            # Train Model
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)

            # Step 5: Evaluation Metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')

            st.write("### Evaluation Metrics")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

            # Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            st.pyplot(plt)

        elif model_type == "Regression":
            model = st.sidebar.selectbox("Select Regression Model", [
                "Linear Regression", "Decision Tree", "Random Forest"
            ])

            if model == "Linear Regression":
                regressor = LinearRegression()
            elif model == "Decision Tree":
                regressor = DecisionTreeRegressor()
            elif model == "Random Forest":
                n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
                regressor = RandomForestRegressor(n_estimators=n_estimators)

            # Train Model
            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_test)

            # Step 5: Evaluation Metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            st.write("### Evaluation Metrics")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.write(f"R-Squared (R²): {r2:.2f}")

            # Visualization: Actual vs Predicted
            st.write("### Actual vs Predicted")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            st.pyplot(plt)

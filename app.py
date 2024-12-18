import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the Streamlit app
st.title("Dynamic Linear Regression App 📊")

# Sidebar for file upload
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Sidebar - Let the user select X and Y columns
    st.sidebar.header("Select Features and Target")
    columns = data.columns.tolist()

    # Select X (Features)
    x_columns = st.sidebar.multiselect("Select Independent Variables (X):", columns)

    # Select Y (Target)
    y_column = st.sidebar.selectbox("Select Dependent Variable (Y):", columns)

    # Check if selections are valid
    if not x_columns or not y_column:
        st.sidebar.error("Please select at least one independent variable (X) and one dependent variable (Y).")
    else:
        # Split data into features (X) and target (Y)
        X = data[x_columns]
        y = data[y_column]

        # Train-Test Split
        test_size = st.sidebar.slider("Test Data Ratio:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display Model Performance Metrics
        st.write("### Model Performance")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**R-squared (R²):** {r2:.4f}")

        # Display Predictions
        st.write("### Predicted vs Actual")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.write(results.head())

        # User Input for Custom Prediction
        st.sidebar.header("Make Predictions")
        input_values = {}
        for col in x_columns:
            input_values[col] = st.sidebar.number_input(f"Enter value for {col}:", value=0.0)

        # Button to make prediction
        if st.sidebar.button("Predict"):
            try:
                input_df = pd.DataFrame([input_values])
                prediction = model.predict(input_df)
                st.sidebar.success(f"Predicted Value: {prediction[0]:,.4f}")
            except Exception as e:
                st.sidebar.error(f"Prediction Error: {e}")

else:
    st.info("📤 Please upload a CSV file to proceed.")

# Footer
st.write("\n---")
st.write("Built with ❤️ using Streamlit | Linear Regression App 📈")

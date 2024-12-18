import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the Streamlit app
st.title("Linear Regression App for Housing Prices 🏠")

# Sidebar instructions
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load dataset
        data = pd.read_csv(uploaded_file)
        st.write("### Preview of Dataset")
        st.write(data.head())

        # Ensure the dataset contains required columns
        required_columns = ["area", "bedrooms", "bathrooms", "stories", "price"]
        if not all(col in data.columns for col in required_columns):
            st.error(f"Error: The dataset must contain the following columns: {', '.join(required_columns)}")
        else:
            # Sidebar for feature selection
            st.sidebar.header("Feature Selection")
            feature_options = ["area", "bedrooms", "bathrooms", "stories"]
            selected_features = st.sidebar.multiselect("Select Features (X):", feature_options, default=feature_options)

            target_column = "price"

            if not selected_features:
                st.error("Please select at least one feature for the model.")
            else:
                # Prepare data for training
                X = data[selected_features]
                y = data[target_column]

                # Split the data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Linear Regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Display performance metrics
                st.write("### Model Performance")
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write(f"**R-squared (R²):** {r2:.2f}")

                # Sidebar inputs for prediction
                st.sidebar.header("Make Predictions")
                input_data = {}
                for feature in selected_features:
                    input_data[feature] = st.sidebar.number_input(f"Enter value for {feature}:", value=0.0)

                # Predict button
                if st.sidebar.button("Predict Price"):
                    try:
                        input_df = pd.DataFrame([input_data])
                        prediction = model.predict(input_df)
                        st.sidebar.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
                    except Exception as e:
                        st.sidebar.error(f"Error in Prediction: {e}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("📤 Please upload a CSV file to get started.")

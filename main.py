import numpy as np
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="NUTRI TRACKER APP")

# Check scikit-learn version
st.write(f"Scikit-learn version: {sklearn.__version__}")

# Load the trained model
loaded_model = pickle.load(open('trained_model (2).sav', 'rb'))

# Function to classify nutritional status
def classify_nutritional_status(prediction):
    try:
        # Convert input data to numpy array and ensure all inputs are numeric
        input_data_as_numpy_array = np.asarray(prediction[1:], dtype=float)
        # Check if gender is 'Girl' or 'Boy' and replace it with 0 or 1 respectively
        gender = 0 if prediction[0] == 'Girl' else 1
        input_data_as_numpy_array = np.insert(input_data_as_numpy_array, 0, gender)
    except ValueError:
        st.error("Please make sure all inputs are numeric.")
        return

    if len(input_data_as_numpy_array) != 4:
        st.error("Please provide all 4 inputs: Gender, Age in months, Height (cm), Weight (kg)")
        return

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make predictions
    try:
        prediction_value = loaded_model.predict(input_data_reshaped)[0]
    except AttributeError as e:
        st.error(f"Model attribute error: {e}")
        return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

    # Determine nutritional status based on prediction
    if 2.8 <= prediction_value <= 3.7:
        return "SEVERE STUNTING"
    elif 4.8 <= prediction_value <= 5.7:
        return "STUNTING"
    elif 3.8 <= prediction_value <= 4.7:
        return "SEVERE WASTING"
    elif 6.8 <= prediction_value <= 7.7:
        return "WASTING"
    elif 5.8 <= prediction_value <= 6.7:
        return "UNDERWEIGHT"
    elif prediction_value == 1:
        return "HEALTHY HEIGHT"
    elif 1.8 <= prediction_value <= 2.7:
        return "HEALTHY HEIGHT"
    else:
        return "UNKNOWN STATUS"


# Load recommendation data
recommend_list = pickle.load(open("recommendations.pkl", 'rb'))
similarity = pickle.load(open("similarity.pkl", 'rb'))

def recommend(typi):
    # Filter recommendations based on the given type
    malnutrition_rows = recommend_list[recommend_list['MALNUTRITION TYPE'] == typi]

    # Check if matching malnutrition type is found
    if malnutrition_rows.empty:
        return []  # Return an empty list if no matching malnutrition type is found

    # Get the index of the matching malnutrition type
    malnutrition_index = malnutrition_rows.index[0]
    distances = similarity[malnutrition_index]

    # Sort distances and get indices of top recommendations
    top_recommendations_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[2:6]

    recommended = []  # Initialize as an empty list
    # Retrieve recommended items using the indices
    for i in top_recommendations_indices:
        recommended.append(recommend_list.iloc[i[0]]['RECOMMENDATIONS'])  # Access 'RECOMMENDATIONS' column
    return recommended

def main():
    st.sidebar.success("User Form")

    page = st.sidebar.radio("Navigation", ["Home", "NUTRI CALCULATOR", "RECOMMENDATION"])

    if page == "Home":
        st.title('NUTRI-TRACKER ')
        html_temp = """
                    <div style="background-color: orange; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #333333; text-align: center;">NUTRI-TRACKER </h2> 
                    </div>
                    """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write('"NOURISHING TODAY\'S CHILDREN ENSURES A THRIVING TOMORROW."')

    elif page == "NUTRI CALCULATOR":
        st.title('NUTRI CALCULATOR')
        html_temp = """
                    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #333333; text-align: center;">NUTRI CALCULATOR</h2> 
                    </div>
                    """
        st.markdown(html_temp, unsafe_allow_html=True)

        gender_option = st.selectbox('Select gender:', [('👧 Girl', 'Girl'), ('👦 Boy', 'Boy')], format_func=lambda x: x[0])

        if gender_option[1] == 'Girl':
            st.write('👧 Girl')
        elif gender_option[1] == 'Boy':
            st.write('👦 Boy')

        def validate_input(age_in_months):
            if not age_in_months.isdigit():
                return False
            age_in_months = int(age_in_months)
            return 0 <= age_in_months <= 60

        # Get user input
        MONTH = st.text_input('Enter age in months(up to 5 years only):')
        HEIGHT = st.text_input('Enter height in cm(up to 120 cm):')
        WEIGHT = st.text_input('Enter weight in kg(up to 21 kg):')

        # Validate age input
        if MONTH:
            if validate_input(MONTH):
                st.success(f"Age: {MONTH} months, Height: {HEIGHT} cm, Weight: {WEIGHT} kg")
                # Proceed with further processing
            else:
                st.error("Please enter a valid age in months (0-60 months).")
        if st.button('MALNUTRITION TYPE'):
            diagnosis = classify_nutritional_status([gender_option[1], MONTH, HEIGHT, WEIGHT])
            if diagnosis:
                st.success(f"Nutritional Status: {diagnosis}")

    elif page == "RECOMMENDATION":
        st.title('RECOMMENDATION SYSTEM OF MALNUTRITION')
        html_temp = """
                    <div style="background-color: #ADD8E6; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #333333; text-align: center;">RECOMMENDATION SYSTEM OF MALNUTRITION</h2> 
                    </div>
                    """
        st.markdown(html_temp, unsafe_allow_html=True)

        selected_type = st.selectbox('Select type for which you want to get recommendations.', (
            'SEVEREWASTING(kg)', 'WASTING(kg)', 'SEVERESTUNTING', 'STUNTING', 'UNDERWEIGHT(kg)', 'HEALTHYHEIGHT(cm)',
            'HEALTHYWEIGHT(kg)'))
        if st.button('RECOMMEND'):
            rec = recommend(selected_type)
            for i in rec:
                st.write(i)

# Check if the script is run directly
if __name__ == '__main__':
    main()

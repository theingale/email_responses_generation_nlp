# Necessary imports
import streamlit as st
from model.model import build_model
from utils.preprocessors import clean_text
from utils.preprocessors import preprocess_text
from utils.prediction import predict




# Application main function
def main():
    """
    Application main function
    
    """
    # Title and Description 
    st.title("Short Response Suggestion for E-mails")
    st.write("An AI application that suggests relevant short response for your Email text.")
    st.text('')

    # Make model ready
    model = build_model()

    # Input text
    input_text = st.text_area("Enter Email Text: (Maximum 50 Words Allowed)",
                               value="Text",
                               height=None, label_visibility="visible")

    # Check input text length
    input_text_length = len(input_text.split())
    suggestion_button_status = True
    
    if input_text_length <= 0:
        st.warning('Text field can not be empty...!', icon="⚠️")

    elif input_text_length > 50:
        st.warning('Text word limit exceeds...!', icon="⚠️")

    else:
        suggestion_button_status = False
    
    suggestion_bt = st.button("Get Response Suggestion", disabled=suggestion_button_status)

    if suggestion_bt:
        
        # Preprocess input_text
        input_text = clean_text(input_text)
        input_sequence = preprocess_text(input_text)

        # Predict response
        predicted_response = predict(input_sequence, model)
        print(predicted_response)

        # Show response
        st.subheader("Machine Suggested Response : ")
        st.write(predicted_response)



if __name__ == '__main__':
    main()
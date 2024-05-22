"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a simple generation or Q&A use case without comprehensive prompt tuning
"""

# Install the wml and streamlit api your Python env prior to running this example:
# pip install ibm-watson-machine-learning
# pip install streamlit

# In non-Anaconda Python environments, you may also need to install dotenv
# pip install python-dotenv

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

import streamlit as st

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# Important: hardcoding the API key in Python code is not a best practice. We are using
# this approach for the ease of demo setup. In a production application these variables
# can be stored in an .env or a properties file

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

# These global variables will be updated in get_credentials() functions
watsonx_project_id = "8c222d58-2bff-4a94-a14e-46b01b02ff0e"
# Replace with your IBM Cloud key
api_key = "Bearer eyJraWQiOiIyMDI0MDUwNTA4MzkiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC02OTIwMDBBS04zIiwiaWQiOiJJQk1pZC02OTIwMDBBS04zIiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiZmRjMzY3OGQtYjg4Yy00YjQ2LTkzZDEtMDBhNmRiNWQ2ZmU0IiwiaWRlbnRpZmllciI6IjY5MjAwMEFLTjMiLCJnaXZlbl9uYW1lIjoiQWNobWFkIFJlemEiLCJmYW1pbHlfbmFtZSI6IkZhaGxldmkiLCJuYW1lIjoiQWNobWFkIFJlemEgRmFobGV2aSIsImVtYWlsIjoiYWNobWFkcmV6YTk5M0BnbWFpbC5jb20iLCJzdWIiOiJhY2htYWRyZXphOTkzQGdtYWlsLmNvbSIsImF1dGhuIjp7InN1YiI6ImFjaG1hZHJlemE5OTNAZ21haWwuY29tIiwiaWFtX2lkIjoiSUJNaWQtNjkyMDAwQUtOMyIsIm5hbWUiOiJBY2htYWQgUmV6YSBGYWhsZXZpIiwiZ2l2ZW5fbmFtZSI6IkFjaG1hZCBSZXphIiwiZmFtaWx5X25hbWUiOiJGYWhsZXZpIiwiZW1haWwiOiJhY2htYWRyZXphOTkzQGdtYWlsLmNvbSJ9LCJhY2NvdW50Ijp7InZhbGlkIjp0cnVlLCJic3MiOiI4NDUyNTlhYzRhMDU0NWQyODg5MThiZWU1Y2MwZjA5YyIsImltc191c2VyX2lkIjoiMTIwNTQ5NjkiLCJmcm96ZW4iOnRydWUsImlzX2VudGVycHJpc2VfYWNjb3VudCI6ZmFsc2UsImVudGVycHJpc2VfaWQiOiI5YTBjMjUwOWE0OTk0N2U0YTFkYjJhYzczZGI4MzI5MyIsImltcyI6IjI4MTEwMDUifSwiaWF0IjoxNzE2MzkwNjgzLCJleHAiOjE3MTYzOTQyODMsImlzcyI6Imh0dHBzOi8vaWFtLmNsb3VkLmlibS5jb20vaWRlbnRpdHkiLCJncmFudF90eXBlIjoidXJuOmlibTpwYXJhbXM6b2F1dGg6Z3JhbnQtdHlwZTphcGlrZXkiLCJzY29wZSI6ImlibSBvcGVuaWQiLCJjbGllbnRfaWQiOiJkZWZhdWx0IiwiYWNyIjoxLCJhbXIiOlsicHdkIl19.YpbW4LLIsKXSDAT48E105zr_AkH7vuLtKr_9eMU0f85Otu4fSdildxy_zziLaofOx9JUIXpE4jEBtndroHJqRlTZb8OgHbTlD2AAO3Wj1p4zV74085c-F_nHKdzgYyDH9RzPuyCYPFQ0P7NpfdIk2TNKT5xoTeAw7s4opR-jUA4lHz_iTjpMjbUJF_xbuXlnAAFwXwaEqaC1kW7IsIRoJ3XXKTnoRwO0_Sr6rjRQLREh1TdlbDM7kUXJgis6t9EEAp1Bs7viAXqghMISFbl5osigzchHJOW-y9B7GiFn1R1JcuGWOnMOQrgLBVk4nP_oZOlKdSxAoKhYGvaGHzcS4g"

def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)

    print("*** Got credentials***")

# The get_model function creates an LLM model object with the specified parameters
def get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )

    return model

def get_prompt(question):

    # Prompts are passed to LLMs as one string. We are building it out as separate strings for ease of understanding
    # Instruction
    instruction = "Answer this question briefly."
    # Examples to help the model set the context
    examples = "\n\nQuestion: What is the capital of Germany\nAnswer: Berlin\n\nQuestion: What year was George Washington born?\nAnswer: 1732\n\nQuestion: What are the main micro nutrients in food?\nAnswer: Protein, carbohydrates, and fat\n\nQuestion: What language is spoken in Brazil?\nAnswer: Portuguese \n\nQuestion: "
    # Question entered in the UI
    your_prompt = question
    # Since LLMs want to "complete a document", we're are giving it a "pattern to complete" - provide the answer
    end_prompt = "Answer:"

    final_prompt = instruction + examples + your_prompt + end_prompt

    return final_prompt

def answer_questions():

    # Set the api key and project id global variables
    get_credentials()

    # Web app UI - title and input box for the question
    st.title('🌠Test watsonx.ai LLM')
    user_question = st.text_input('Ask a question, for example: What is IBM?')

    # If the quesiton is blank, let's prevent LLM from showing a random fact, so we will ask a question
    if len(user_question.strip())==0:
        user_question="What is IBM?"

    # Get the prompt
    final_prompt = get_prompt(user_question)

    # Display our complete prompt - for debugging/understanding
    print(final_prompt)

    # Look up parameters in documentation:
    # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
    model_type = ModelTypes.FLAN_UL2
    max_tokens = 100
    min_tokens = 20
    decoding = DecodingMethods.GREEDY
    stop_sequences = ['.']

    # Get the model
    model = get_model(model_type, max_tokens, min_tokens, decoding,stop_sequences)

    # Generate response
    generated_response = model.generate(prompt=final_prompt)
    model_output = generated_response['results'][0]['generated_text']
    # For debugging
    print("Answer: " + model_output)

    # Display output on the Web page
    formatted_output = f"""
        **Answer to your question:** {user_question} \
        *{model_output}*</i>
        """
    st.markdown(formatted_output, unsafe_allow_html=True)

# Invoke the main function
answer_questions()

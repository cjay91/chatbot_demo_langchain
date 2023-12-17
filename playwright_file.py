from langchain.document_loaders import PlaywrightURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
import openai
import streamlit as st
from streamlit_extras.colored_header import colored_header

# client = OpenAI()


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
# past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')


# User input
# Function for taking user provided prompt as input
user_input = st.chat_input("Ask Something")

urls = [
    'https://www.dilmahtea.com/all-about-tea/facts-of-tea'
]

loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)

docs = text_splitter.split_documents(data)


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

db = FAISS.from_documents(docs,embeddings)

db.save_local("faiss_index")

new_db = FAISS.load_local("faiss_index",embeddings)


template = '''
User:
You are an expert in your domain. Please provide your expert and user-friendly response based on the context provided. You should aim to provide a clear, concise, and accurate response including contact details. If the question is not taken from the given context, do not respond with general knowledge, and only leave a polite message. Polite message should be this - Thank you for your inquiry. Please note that the information provided by this program is restricted to Dilmah Tea Company information. If you have inquiries related to Dilmah or need assistance on a different topic within the defined scope, feel free to ask. I'm here to help!

For optimal assistance, kindly formulate your questions using keywords related to the Dilmah Tea domain.

Modified Instructions:

1. Craft responses that are clear, concise, and accurate based on the given context.

2. Include contact details in your responses where applicable.

3. If a question is not taken from the given context, do not respond with general knowledge. Instead, use the predefined message provided below.

4. Encourage users to formulate questions using keywords related to the Agroworld domain for optimal assistance.

5. Ensure that your responses align with the user's request for expert and user-friendly information within the defined domain.

6. Response with a generic greeting if the user greets. No need to tell about Dilmah

7. Ignore simple deviation of the company name

Modified Instructions:

Thank you for your inquiry. Please note that the information provided by this program is restricted to Dilmah Tea Company information. If you have inquiries related to Dilmah or need assistance on a different topic within the defined scope, feel free to ask. I'm here to help!

'''

# llm = ChatOpenAI(temperature = 0.0,
#                  model_name = "gpt-4"
#                 )
                 

# llm_chain = LLMChain(llm=llm, prompt=chat_template)

# def ask(user_query):
#     docs = new_db.similarity_search(user_query)
#     res = llm_chain.predict(context = docs, user_input = user_query )
#     return res


docs = new_db.similarity_search(user_input)

myMessages = []
myMessages.append(
    {"role": "system", "content":template })

myMessages.append(
    {"role": "user", "content": "context:\n\n{}.\n\n Answer the following user query according to the given context.:\nuser_input: {}".format(docs, user_input)})


if user_input:
    # st.session_state.messages.append({"role": "user", "content": "context:\n\n{}.\n\n Answer the following user query according to the given context:\nuser_input: {}".format(context, user_input)})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-4",
            messages=myMessages,
            stream=True,
        ):
            if "delta" in response["choices"][0] and "content" in response["choices"][0]["delta"]:
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
st.session_state.messages.append({"role": "user", "content": user_input})
st.session_state.messages.append({"role": "assistant", "content": full_response})
    

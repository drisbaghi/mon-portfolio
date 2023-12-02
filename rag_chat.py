
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import WebBaseLoader

URLS = [
    'https://lscosmetics.ma/',
    'https://baraboucle.com/',
    'https://www.lexpress.fr/',
    'https://www.ducray.com/',
    'https://dreamact.eu/',
    'https://aunea-cosmetique.fr/',
    'https://lilikiwi.fr/fr/blog/comment-choisir-un-shampoing-pour-enfants--n25',
    'https://cheveux.ooreka.fr/',
    'https://madame.lefigaro.fr/',
    'https://www.nuoobox.com/',
    'https://www.kryneo.com/',
    'https://volna-cosmetiques.fr/',
    'https://clean-sponge.fr/'
]

loader = WebBaseLoader(URLS)
data = loader.load()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
websites_data = text_splitter.split_documents(data)

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(deployment="embeddings-cosmetics", embedding_ctx_length=9000)

from langchain.vectorstores import Chroma

websearch = Chroma.from_documents(websites_data, embeddings)

from langchain.chat_models import ChatOpenAI

model = ChatOpenAI(model='gpt-4',temperature=0)

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

memory = ConversationBufferMemory(
    llm=model, memory_key="chat_history")

template = """You are an assistant for question-answering tasks and marketing agent for our website https://lscosmetics.ma/ in the field of cosmetics,
-Use the following pieces of context to answer the question at the end. 
-You should answer the user in his language either English, Frensh or Moroccan arabic.
-Offer only products from our site: https://lscosmetics.ma/
-All product details, including compositions and descriptions, are available on the LS Cosmetics website,So you can answer questions about that from our website:https://lscosmetics.ma/
-Don't forget to ask the user in the right moment about his contact. 
-Only if the user is ready for product purchasing, lead him to our website: https://lscosmetics.ma/
{context}
Question: {question}
Helpful answer closely linked to the question:"""

rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain = (
    {"context": websearch.as_retriever() , "question": RunnablePassthrough()}
    | rag_prompt_custom
    | model
    | StrOutputParser()
)


import streamlit as st

st.title("LS Cosmetics")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Marhaban"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response =rag_chain.invoke(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})






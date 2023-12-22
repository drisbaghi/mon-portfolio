

import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import WebBaseLoader


URLS = [
    'https://lscosmetics.ma/',
    'https://baraboucle.com/blogs/journal/quel-produit-pour-mon-enfant',
    'https://www.lexpress.fr/styles/beaute/ma-fille-hurle-avant-meme-que-la-brosse-atteigne-ses-cheveux_1243047.html',
    'https://www.ducray.com/fr-be/etats-pelliculaires/symptomes/pellicules-enfant',
    'https://dreamact.eu/fr/blog/article/162/huiles-vegetales',
    'https://aunea-cosmetique.fr/comment-prendre-soin-des-cheveux-crepus-des-enfants-au-quotidien/',
    'https://lilikiwi.fr/fr/blog/comment-choisir-un-shampoing-pour-enfants--n25',
    'https://cheveux.ooreka.fr/astuce/voir/645153/proteger-vos-cheveux-de-la-piscine',
    'https://madame.lefigaro.fr/beaute/ces-erreurs-que-lon-fait-tous-avec-notre-shampoing-071220-194052',
    'https://www.nuoobox.com/blogs/conseils/cosmetiques-bio-conventionnels-differences',
    'https://www.kryneo.com/blogs/conseils-capillaires/les-7-types-d-ingredients-a-eviter-dans-les-produits-capillaires',
    'https://volna-cosmetiques.fr/routine-cheveux-enfants-ideale/',
    'https://clean-sponge.fr/blogs/soin-des-cheveux/conseils-pour-choisir-des-produits-capillaires-vegan-et-ecologiques'
    ]


loader = WebBaseLoader(URLS)
data = loader.load()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=20000, chunk_overlap=100)
websites_data = text_splitter.split_documents(data)

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(deployment="embeddings-cosmetics")

from langchain.vectorstores import Chroma

websearch = Chroma.from_documents(websites_data, embeddings)

from langchain.chat_models import ChatOpenAI

model = ChatOpenAI(model='gpt-3.5-turbo-16k',temperature=0)

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

memory = ConversationBufferMemory(
    llm=model, memory_key="chat_history")

template = """You are an assistant for question-answering tasks and marketing agent for our website https://lscosmetics.ma/ in the field of cosmetics,and you must consider the following points:
-Use the following pieces of context to answer the question at the end.
-You must includ in only your first message the asking of the user about his contact.
-You must answer the user in his language either English, Moroccan arabic (Moroccan Dialect) or Frensh.
-Only recommend products available on the site : https://lscosmetics.ma/.
-You must display our website product pictures in case the user ask for them.
-All product details, including compositions and descriptions, are available on the LS Cosmetics website,So you can answer questions about that from our website:https://lscosmetics.ma/
-You must adjust to the user's gender based on the context, questions, name, and feelings.
-You must say that we help over 5000 moms with their children's hair management routines  only in the right moment as a proof.
-Only if the user is ready for product purchasing, lead him to our website: https://lscosmetics.ma/
{context}
Question: {question}
"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain = (
    {"context": websearch.as_retriever(search_type="similarity"), "question": RunnablePassthrough()}
    | rag_prompt_custom
    | model
    | StrOutputParser()
)

import streamlit as st
import time
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Streamlit app
st.title("LS Cosmetics")

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

    # Assuming rag_chain.invoke returns a response based on user input
    response = rag_chain.invoke(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Countdown
    ph = st.empty()
    N = 20
    for secs in range(N, 0, -1):
        ss = secs % 60
        ph.metric("Countdown", f"{ss:02d}")
        time.sleep(1)

    # Display inactivity message when the countdown reaches 0
    template2 = """Vous êtes le même assistant, donc vous devez utiliser le même langage que l'utilisateur. Parlez à l'utilisateur et demandez-lui la raison de son inactivité dans la conversation."""
    rag_prompt_custom2 = PromptTemplate.from_template(template2)
    rag_chain2 = (
        rag_prompt_custom2
        | model
    )

    ph.metric("Countdown", "00")
    inp=""
    with st.chat_message("assistant"):
        st.markdown(rag_chain2.invoke(inp))








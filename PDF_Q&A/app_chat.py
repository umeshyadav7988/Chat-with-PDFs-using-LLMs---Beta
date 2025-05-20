import streamlit as st
import os
from constants import search_number_messages
from langchain_utils import initialize_chat_conversation
from search_indexing import download_and_index_pdf
import re

def remove_url(url_to_remove):
    if url_to_remove in st.session_state.urls:
        st.session_state.urls.remove(url_to_remove)


# Page title
st.set_page_config(page_title='Talk with PDFs using LLMs - Beta')
st.title('Talk with PDFs using LLMs - (Beta)')

# Initialize session state variables
st.session_state.setdefault('faiss_index', {'indexed_urls': [], 'index': None})
st.session_state.setdefault('conversation_memory', None)
st.session_state.setdefault('messages', [])
st.session_state.setdefault('urls', [])

# Sidebar input for API key and URL submission
with st.sidebar:
    openai_api_key = st.text_input('Step 1 - OpenAI API Key:', type='password')

    with st.form('urls-form', clear_on_submit=True):
        url = st.text_input('Step 2 - URLs to relevant PDFs:')
        add_url_button = st.form_submit_button('Add')
        if add_url_button and url not in st.session_state.urls:
            st.session_state.urls.append(url)

    if st.session_state.urls:
        st.header('URLs added:')
        for url in st.session_state.urls:
            st.write(url)
            st.button(label='Remove', key=f"Remove {url}", on_click=remove_url, kwargs={'url_to_remove': url})
            st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    role = "You" if message["role"] == "user" else "Assistant"
    st.markdown(f"**{role}:** {message['content']}")

# User input
query_text = st.text_input("Your message")
if query_text:
    if not openai_api_key:
        st.error("Please provide your OpenAI API key.")
    else:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        st.markdown(f"**You:** {query_text}")
        st.session_state.messages.append({"role": "user", "content": query_text})

        session_urls = st.session_state.urls

        if st.session_state['faiss_index']['index'] is None or set(st.session_state['faiss_index']['indexed_urls']) != set(session_urls):
            st.session_state['faiss_index']['indexed_urls'] = session_urls
            with st.spinner('Downloading and indexing PDFs...'):
                faiss_index = download_and_index_pdf(session_urls)
                st.session_state['faiss_index']['index'] = faiss_index
        else:
            faiss_index = st.session_state['faiss_index']['index']

        if st.session_state['conversation_memory'] is None:
            conversation = initialize_chat_conversation(faiss_index)
            st.session_state['conversation_memory'] = conversation
        else:
            conversation = st.session_state['conversation_memory']

        user_messages_history = [
            message['content']
            for message in st.session_state.messages[-search_number_messages:]
            if message['role'] == 'user'
        ]
        user_messages_history = '\n'.join(user_messages_history)

        with st.spinner('Querying OpenAI GPT...'):
            try:
                response = conversation.predict(input=query_text, user_messages_history=user_messages_history)
            except Exception as e:
                st.error(f"Error from OpenAI: {str(e)}")
                response = "Sorry, an error occurred while processing your request."

        st.markdown(f"**Assistant:** {response}")
        snippet_memory = conversation.memory.memories[1]
        for page_number, snippet in zip(snippet_memory.pages, snippet_memory.snippets):
            with st.expander(f'Snippet from page {page_number + 1}'):
                snippet = re.sub("<START_SNIPPET_PAGE_\\d+>", '', snippet)
                snippet = re.sub("<END_SNIPPET_PAGE_\\d+>", '', snippet)
                st.markdown(snippet)

        st.session_state.messages.append({"role": "assistant", "content": response})

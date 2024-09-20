from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
import chainlit as cl
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Set up API key for OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    
    # Define RAG prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable AI engineer who's good at explaining stuff like ELI5."
            ),
            ("human", "{context}\n\nQuestion: {question}")
        ]
    )

    # Load documents and create retriever
    ai_framework_document = PyMuPDFLoader(file_path="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf").load()
    ai_blueprint_document = PyMuPDFLoader(file_path="https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf").load()

    # Semantic chunking
    text_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-large"))

    def metadata_generator(document, name):
        fixed_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        collection = fixed_text_splitter.split_documents(document)
        for doc in collection:
            doc.metadata["source"] = name
        return collection

    recursive_framework_document = metadata_generator(ai_framework_document, "AI Framework")
    recursive_blueprint_document = metadata_generator(ai_blueprint_document, "AI Blueprint")
    combined_documents = recursive_framework_document + recursive_blueprint_document

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Vector store and retriever
    vectorstore = Qdrant.from_documents(
        documents=combined_documents,
        embedding=embeddings,
        location=":memory:",
        collection_name="AI Policy"
    )
    
    retriever = vectorstore.as_retriever()
    
    # Set the retriever and prompt into session for reuse
    cl.user_session.set("runnable", model)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("prompt_template", prompt)


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    
    # Define RAG prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable AI engineer who's good at explaining stuff like ELI5."
            ),
            ("human", "{context}\n\nQuestion: {question}")
        ]
    )

    # Load documents and create retriever
    ai_framework_document = PyMuPDFLoader(file_path="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf").load()
    ai_blueprint_document = PyMuPDFLoader(file_path="https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf").load()

    # Semantic chunking
    text_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-large"))

    def metadata_generator(document, name):
        fixed_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        collection = fixed_text_splitter.split_documents(document)
        for doc in collection:
            doc.metadata["source"] = name
        return collection

    recursive_framework_document = metadata_generator(ai_framework_document, "AI Framework")
    recursive_blueprint_document = metadata_generator(ai_blueprint_document, "AI Blueprint")
    combined_documents = recursive_framework_document + recursive_blueprint_document

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Vector store and retriever
    vectorstore = Qdrant.from_documents(
        documents=combined_documents,
        embedding=embeddings,
        location=":memory:",
        collection_name="AI Policy"
    )
    
    retriever = vectorstore.as_retriever()
    
    # Set the retriever and prompt into session for reuse
    cl.user_session.set("runnable", model)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("prompt_template", prompt)

@cl.on_message
async def on_message(message: cl.Message):
    # Get the stored model, retriever, and prompt
    model = cast(ChatOpenAI, cl.user_session.get("runnable"))  # type: ChatOpenAI
    retriever = cl.user_session.get("retriever")  # Get the retriever from the session
    prompt_template = cl.user_session.get("prompt_template")  # Get the RAG prompt template

    # Log the message content
    print(f"Received message: {message.content}")

    # Retrieve relevant context from documents based on the user's message
    relevant_docs = retriever.get_relevant_documents(message.content)
    print(f"Retrieved {len(relevant_docs)} documents.")

    if not relevant_docs:
        print("No relevant documents found.")
        await cl.Message(content="Sorry, I couldn't find any relevant documents.").send()
        return

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Log the context to check
    print(f"Context: {context}")

    # Construct the final RAG prompt
    final_prompt = prompt_template.format(context=context, question=message.content)
    print(f"Final prompt: {final_prompt}")

    # Initialize a streaming message
    msg = cl.Message(content="")

    # Stream the response from the model
    async for chunk in model.astream(
        final_prompt,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # Extract the content from AIMessageChunk and concatenate it to the message
        await msg.stream_token(chunk.content)

    await msg.send()

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import GROQ_MODEL, CHROMA_DIR, EMBED_MODEL, require_groq_api_key


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def _doc_name(doc):
    return doc.metadata.get("filename") or doc.metadata.get("source") or "unknown"


def format_docs(docs):
    return "\n\n".join([f"# {_doc_name(doc)}\n{doc.page_content}" for doc in docs])


def _get_llm(temperature):
    return ChatGroq(
        api_key=require_groq_api_key(),
        model=GROQ_MODEL,
        temperature=temperature,
    )


def get_qa_chain():
    llm = _get_llm(temperature=0.2)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=""" You are an expert code analyst. Use the code below to answer the question.
        Always mention which files the answer relates to.

        Code context:
        {context}

        Question: {question}

        Answer:"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def get_doc_chain(filename: str):
    llm = _get_llm(temperature=0.1)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8, "filter": {"filename": filename}}
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a technical writer. Generate clean markdown documents for the following code file.
        Include: purpose of the file, all functions with their parameters and what they return, and a usage example

        Code:
        {context}

        File: {question}

        Documentation:"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

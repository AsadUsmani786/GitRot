from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import GROQ_API_KEY,GROQ_MODEL 





CHROMA_DIR= "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name = EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

def format_docs(docs):
    return "\n\n".join([f"# {doc.metadata['filename']}\n{doc.page_content}" for doc in docs])

def get_qa_chain():
    llm = ChatGroq(api_key=GROQ_API_KEY, model = GROQ_MODEL, temperature=0.2)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k":5})

    prompt = PromptTemplate(
        input_variable=["context","question"],
        template = """ You are an expert code analyst. Use the code below to answer the question.
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


if __name__ == "__main__":
    chain,retriever = get_qa_chain()
    question = "what does the load_code_files function do?"
    answer =chain.invoke(question)
    sources = retriever.invoke(question)
    print("Answer:", answer)
    print("\nSources:", [doc.metadata["filename"] for doc in sources])

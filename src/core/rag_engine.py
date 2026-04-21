from typing import List, Tuple, Generator

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from src.config import LLM_CONFIG, RETRIEVER_CONFIG

def setup_hybrid_retriever(vector_store, documents=mock_docs):
    # Khởi tạo BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3
    
    # Khởi tạo FAISS/Vector Retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Kết hợp bằng EnsembleRetriever (Tỷ lệ 50-50)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def run_rag_chain(user_question, vector_store, chat_history=""):
    # 1. Khởi tạo LLM
    llm = OllamaLLM(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],  
        top_p=LLM_CONFIG["top_p"],        
        repeat_penalty=LLM_CONFIG["repeat_penalty"]
    )

    # 2. Logic tạo Prompt đa ngôn ngữ
    def get_prompt_template(user_input: str):
        vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        is_vietnamese = any(char in user_input.lower() for char in vietnamese_chars)
        
        if is_vietnamese:
            template = """Sử dụng ngữ cảnh và lịch sử trò chuyện dưới đây để trả lời câu hỏi.
            Nếu bạn không biết, chỉ cần nói là bạn không biết.
            Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.
            
            Lịch sử trò chuyện gần đây:
            {chat_history}
            
            Ngữ cảnh: {context}
            
            Câu hỏi: {question}
            
            Trả lời:"""
        else:
            template = """Use the following context and conversation history to answer the question.
            If you don't know the answer, just say you don't know.
            Keep answer concise (3-4 sentences).
            
            Recent Conversation History:
            {chat_history}
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
        return PromptTemplate(
            template=template,
            input_variables=["context", "question", "chat_history"]
        )

    prompt = get_prompt_template(user_question)
    
    # 3. Cấu hình Retriever
    retriever = setup_hybrid_retriever(vector_store)
    
    # Lấy ra danh sách các đoạn text chứa câu trả lời (kèm metadata)
    source_docs = retriever.invoke(user_question)
    
    # Nối text lại để làm ngữ cảnh cho AI
    context_text = "\n\n".join(doc.page_content for doc in source_docs)
    
    # 4. Khởi tạo Pipeline chỉ chứa Prompt và LLM
    chain = prompt | llm | StrOutputParser()

    response_stream = chain.stream(
        {
            "context": context_text,
            "question": user_question,
            "chat_history": chat_history,
        }
    )

    return response_stream, reranked_docs, unique_sources
import streamlit as st
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder
from src.config.settings import LLM_CONFIG, RERANKER_CONFIG


@st.cache_resource(show_spinner=False)
def get_llm():
    """Khởi tạo LLM một lần duy nhất, dùng chung cho toàn bộ hệ thống"""
    return OllamaLLM(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        top_p=LLM_CONFIG["top_p"],
        repeat_penalty=LLM_CONFIG["repeat_penalty"],
    )


@st.cache_resource(show_spinner=False)
def get_reranker():
    """Khởi tạo Cross-Encoder Re-ranker"""
    return CrossEncoder(RERANKER_CONFIG["model_name"])

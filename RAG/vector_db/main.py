import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
import warnings
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
import torch
from langchain.schema import Document, BaseRetriever
from pydantic import BaseModel, Field
from typing import List, Any
import logging
import gc
from search_web import test_search
from peft import PeftModel, PeftConfig

# Cấu hình logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        config_path = os.path.normpath(config_path)
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_model(api_key, model_type="gpt-4o-mini"):
    try:
        if model_type == "gpt-4o-mini":
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.5,
                openai_api_key=api_key
            )
        elif model_type == "llama":
            # Dọn bộ nhớ GPU nếu có
            if torch.cuda.is_available():
                clear_gpu_memory()
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Lấy đường dẫn đến thư mục models
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adapter_path = os.path.join(current_dir, "..", "Fine_tuning", "models")
            
            # Load cấu hình PEFT và lấy tên mô hình cơ sở
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # Load mô hình cơ sở
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load adapter weights
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Tạo pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=False,
                truncation=True
            )
            
            # Tạo một class wrapper để match với interface của ChatOpenAI
            class LlamaWrapper:
                def __init__(self, pipe):
                    self._pipe = pipe
                
                def clean_gpu_memory(self):
                    clear_gpu_memory()
                
                def __call__(self, prompt):
                    return self._pipe(prompt)[0]['generated_text']
            
            return LlamaWrapper(pipe)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_vector_db(api_key, collection_name):
    try:
        config = load_config()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persist_directory = os.path.join(current_dir, "chroma_db")
        
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        collection_path = os.path.join(persist_directory, collection_name)
        if not os.path.exists(collection_path):
            os.makedirs(collection_path)
            
        qa_path = os.path.join(persist_directory, "Q_A")
        if not os.path.exists(qa_path):
            os.makedirs(qa_path)
        
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        
        domain_db = Chroma(
            persist_directory=collection_path,
            embedding_function=embedding_model
        )
        
        qa_db = Chroma(
            persist_directory=qa_path,
            embedding_function=embedding_model
        )
        
        return domain_db, qa_db
    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        raise

def classify_question_domain(question):
    domains = {
        "thong_tin_chung": ["thông tin chung", "giới thiệu", "tổng quan", "trường", "bách khoa"],
        "de_an_tuyen_sinh": ["đề án", "chỉ tiêu", "phương thức"],
        "xet_tuyen_tai_nang": ["tài năng", "xét tuyển tài năng", "năng khiếu"],
        "diem_chuan_tuyen_sinh": [
            "điểm chuẩn", "điểm trúng tuyển", "điểm đầu vào", 
            "điểm thi", "điểm xét tuyển",
            "điểm chuẩn 2024", "điểm trúng tuyển 2024",
            "năm 2024", "2024"
        ],
        "ky_thi_danh_gia_tu_duy": ["tư duy", "kỳ thi", "đánh giá", "ĐGTD", "thi tư duy"],
        "xac_thuc_chung_chi_ngoai_ngu": ["ngoại ngữ", "chứng chỉ", "tiếng anh", "xác thực"],
        "huong_nghiep": ["hướng nghiệp", "nghề nghiệp", "định hướng", "ngành học"]
    }
    
    question = question.lower()
    for domain, keywords in domains.items():
        if any(keyword in question for keyword in keywords):
            return domain
    return "Q_A"

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
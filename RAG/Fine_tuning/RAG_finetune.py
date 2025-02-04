from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import os
import json
import logging
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch
from search_web import test_search
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_db.model import (
    classify_question_domain,
    load_vector_db,
    get_relevant_chunks,
    CustomRetriever,
    create_qa_chain
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        
def load_finetuned_model():
    try:
        # Clear GPU memory first
        clear_gpu_memory()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models")
        
        if not os.path.exists(model_path):
            raise Exception(f"Model directory not found at {model_path}")
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load model with new quantization config
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.5,
            max_length=2048,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        return HuggingFacePipeline(pipeline=pipe)
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Reuse other functions from model.py
def create_prompt():
    template = """Bạn là giáo viên phòng tư vấn tuyển sinh của trường Đại học Bách Khoa Hà Nội.
    Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn, truyền tải đầy đủ thông tin bạn nhận được.
    Hãy nói với vai trò là một thầy cô giáo trong trường, sử dụng ngôn ngữ thân thiện và dễ hiểu.
    
    Nếu không tìm thấy thông tin trong context hoặc thông tin không đầy đủ, hãy trả lời:
    "Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question", "web_url"])
    return prompt

# Reuse classify_question_domain, load_vector_db, get_relevant_chunks, CustomRetriever from model.py

def get_answer(question):
    try:
        # Clear GPU memory before processing new question
        clear_gpu_memory()
        
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        
        if not api_key:
            raise Exception("OpenAI API key not found in config.json")
        
        domain = classify_question_domain(question)
        model = load_finetuned_model()
        domain_db, qa_db = load_vector_db(api_key, domain)
        web_url = test_search(question)
        prompt = create_prompt()
        qa_chain = create_qa_chain(domain_db, qa_db, model, prompt, web_url)
        
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", []),
            "domain": domain
        }
        
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        raise

if __name__ == "__main__":
    while True:
        question = input("\nCâu hỏi: ")
        if question.lower() in ['quit', 'q', 'exit']:
            break
            
        try:
            result = get_answer(question)
            print("\nCâu trả lời:", result["answer"])
            
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Error in main loop: {str(e)}")

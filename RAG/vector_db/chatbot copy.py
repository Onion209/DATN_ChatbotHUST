import os
import torch
import logging
import sys
from typing import List, Tuple, Set
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_db.model import (
    classify_question_domain,
    load_vector_db,
    get_relevant_chunks,
    load_config
)
from search_web import test_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    # ... existing ModelManager code ...
    _instance = None
    _pipe = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance
    
    def __init__(self):
        if self._pipe is not None:
            return
        self._pipe = self.load_fine_tuned_model()

    @staticmethod
    def clean_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleaned GPU memory")

    def load_fine_tuned_model(self):
        # ... existing load_fine_tuned_model code ...
        try:
            if torch.cuda.is_available():
                self.clean_gpu_memory()
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adapter_path = os.path.join(current_dir, "models")
            
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=2048,  # Tăng từ 512 lên 2048 hoặc cao hơn
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=False,
                truncation = True
            )
            
            return pipe
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise

def get_data_with_context(question: str) -> Tuple[str, str, Set[str]]:
    """Lấy dữ liệu từ database kèm context mở rộng"""
    try:
        # 1. Khởi tạo và cấu hình
        config = load_config()
        api_key = config.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found in config")

        # 2. Phân loại domain và tìm kiếm web
        domain = classify_question_domain(question)
        web_url = test_search(question)
        
        # 3. Load vector databases
        domain_db, qa_db = load_vector_db(api_key, domain)
        
        # 4. Lấy chunks với relevance scores
        documents = get_relevant_chunks(question, domain_db, qa_db, k=3)
        
        if not documents:
            return None, web_url, None

        # 5. Sắp xếp chunks theo score
        documents.sort(key=lambda x: float(x.metadata.get('score', 0)), reverse=True)
        
        # 6. Lấy context mở rộng cho mỗi chunk
        expanded_context_parts = []
        sources = set()
        
        for i, doc in enumerate(documents):
            # Lấy metadata của chunk hiện tại
            current_metadata = doc.metadata
            source = current_metadata.get('source')
            page = current_metadata.get('page', 'Unknown')
            
            # Thêm source vào danh sách
            if source and source != 'Unknown':
                source_info = f"{source} (trang {page})" if page != 'Unknown' else source
                sources.add(source_info)
            
            # Tạo context mở rộng bao gồm chunk trước và sau
            context_group = []
            
            # Thêm chunk trước nếu có
            if i > 0:
                context_group.append(f"Đoạn trước: {documents[i-1].page_content}")
            
            # Thêm chunk hiện tại
            context_group.append(f"Đoạn chính: {doc.page_content}")
            
            # Thêm chunk sau nếu có
            if i < len(documents) - 1:
                context_group.append(f"Đoạn sau: {documents[i+1].page_content}")
            
            expanded_context_parts.append("\n".join(context_group))

        # 7. Tạo context tổng hợp
        context = "\n\n".join(expanded_context_parts)
        
        return context, web_url, sources

    except Exception as e:
        logger.error(f"Error in get_data_with_context: {str(e)}")
        return None, None, None

def generate_response(prompt: str) -> str:
    try:
        model_manager = ModelManager()
        model_manager.clean_gpu_memory()
        
        pipe = model_manager._pipe
        response = pipe(prompt)[0]['generated_text']
        
        model_manager.clean_gpu_memory()
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def create_prompt_llama(question: str, context: str) -> str:
    return f"""Bạn là một giáo viên phòng tư vấn tuyển sinh đại học Bách Khoa Hà Nội.
Bạn có nhiệm vụ tư vấn tuyển sinh cho người dùng và trả lời đúng những câu hỏi mà người dùng đặt ra.
Trả lời câu hỏi một người dùng thân thiện, chính xác. Nếu không tìm được thông tin hãy trả lời 'Vui lòng tra thêm thông tin trên website: 'https://ts.hust.edu.vn/'.

### Instruction:
{question}

### Input:
{context}

### Response:
"""

def main():
    try:
        while True:
            question = input("\nNhập câu hỏi (hoặc 'q' để thoát): ")
            if question.lower() == 'q':
                break
            
            # Lấy dữ liệu và context mở rộng
            context, web_url, sources = get_data_with_context(question)
            
            if context is None:
                print(f"\nKhông tìm thấy thông tin phù hợp. Bạn có thể tham khảo thêm tại: {web_url}")
                continue
            
            # Tạo prompt và generate câu trả lời
            prompt = create_prompt_llama(question, context)
            response = generate_response(prompt)
            
            # In kết quả
            print("\nCâu trả lời:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            if sources:
                print("\nNguồn tham khảo:")
                for source in sources:
                    print(f"- {source}")
            
            if web_url:
                print(f"\nTham khảo thêm tại: {web_url}")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main() 
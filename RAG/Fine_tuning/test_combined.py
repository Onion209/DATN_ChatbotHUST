import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import sys
import json
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_db.model import (
    classify_question_domain,
    load_vector_db,
    get_relevant_chunks
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
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
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=False,
                truncation=True
            )
            
            return pipe
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def load_config():
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(current_dir, "vector_db", "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def create_prompt(context: str, question: str) -> str:
    return f"""Bạn là giáo viên phòng tư vấn tuyển sinh của trường Đại học Bách Khoa Hà Nội.
    Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn, truyền tải đầy đủ thông tin bạn nhận được.
    Hãy nói với vai trò là một thầy cô giáo trong trường, sử dụng ngôn ngữ thân thiện và dễ hiểu.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """

def test_combined(question: str):
    try:
        # Load config và khởi tạo
        config = load_config()
        api_key = config.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found in config")

        # Phân loại domain và lấy chunks
        domain = classify_question_domain(question)
        print(f"\nDomain được phân loại: {domain}")
        
        domain_db, qa_db = load_vector_db(api_key, domain)
        print("Đã load vector databases")
        
        chunks = get_relevant_chunks(question, domain_db, qa_db)
        if not chunks:
            return "Xin lỗi em, thông tin này hiện tại cô/thầy chưa có..."

        # Tạo context từ chunks
        context_parts = []
        sources = set()
        
        for chunk in chunks:
            context_parts.append(chunk.page_content)
            source = chunk.metadata.get('source')
            page = chunk.metadata.get('page')
            if source and source != 'Unknown':
                if page and page != 'Unknown':
                    sources.add(f"{source} (trang {page})")
                else:
                    sources.add(source)

        # Tạo prompt và sinh câu trả lời
        context = "\n".join(context_parts)
        prompt = create_prompt(context, question)
        
        model_manager = ModelManager()
        model_manager.clean_gpu_memory()
        
        response = model_manager._pipe(prompt)[0]['generated_text']
        model_manager.clean_gpu_memory()
        
        # Format câu trả lời với nguồn
        answer = response.strip()
        if sources:
            answer += "\n\nThông tin được trích từ: " + ", ".join(sources)
            
        return answer

    except Exception as e:
        logger.error(f"Error in test_combined: {str(e)}")
        return f"Lỗi: {str(e)}"

if __name__ == "__main__":
    try:
        print("Initializing model...")
        model_manager = ModelManager()
        print("Model initialized successfully!")
        
        while True:
            question = input("\nHỏi (hoặc 'q' để thoát): ")
            if question.lower() == 'q':
                break
            
            result = test_combined(question)
            print(f"\nTrả lời: {result}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error in main loop: {str(e)}") 
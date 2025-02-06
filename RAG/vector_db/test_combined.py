import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import sys
import json
import gc
from langchain.prompts import PromptTemplate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_db.model import (
    classify_question_domain,
    load_vector_db,
    get_relevant_chunks
)
from search_web import test_search

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

def create_prompt_combined(context: str, question: str, web_url: str = None) -> str:
    template = """Bạn là giáo viên phòng tư vấn tuyển sinh của trường Đại học Bách Khoa Hà Nội.
    Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và có cấu trúc, theo format sau:
    1. Chỉ trả lời những thông tin liên quan trực tiếp đến câu hỏi
    2. Nếu là điểm chuẩn, cần nêu rõ: điểm số, mã ngành, tên ngành và phương thức xét tuyển
    3. Thông tin phải được trích dẫn từ context được cung cấp
    4. Không thêm thông tin không có trong context
    
    Nếu không tìm thấy thông tin trong context hoặc thông tin không đầy đủ, hãy trả lời:
    "Xin lỗi, hiện tại tôi không có thông tin chính xác về điều này. Tuy nhiên, bạn có thể tham khảo thông tin mới nhất tại: {web_url}"
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question", "web_url"]
    )
    return prompt.format(context=context, question=question, web_url=web_url)

def test_combined(question: str):
    try:
        # 1. Khởi tạo và cấu hình
        config = load_config()
        api_key = config.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found in config")

        # 2. Phân loại domain và tìm kiếm web
        domain = classify_question_domain(question)
        print(f"\nDomain được phân loại: {domain}")
        web_url = test_search(question)
        
        # 3. Load vector databases
        domain_db, qa_db = load_vector_db(api_key, domain)
        print("Đã load vector databases")
        
        # 4. Lấy chunks với relevance scores (k=3 mặc định)
        documents = get_relevant_chunks(question, domain_db, qa_db)
        
        # 5. Xử lý trường hợp không có chunks
        if not documents:
            fallback_message = (
                f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
            )
            return fallback_message

        # 6. Tạo context từ documents và lọc theo độ tin cậy
        context_parts = []
        sources = set()
        
        for doc in documents:
            score = float(doc.metadata.get('score', 0))
            if score >= 0.7:  # Chỉ lấy những document có độ tin cậy từ 0.7 trở lên
                context_parts.append(doc.page_content)
                metadata = doc.metadata
                source_info = []
                
                source = metadata.get('source')
                page = metadata.get('page')
                
                if source and source != 'Unknown':
                    if page and page != 'Unknown':
                        source_info.append(f"{source} (trang {page})")
                    else:
                        source_info.append(source)
                
                if source_info:
                    sources.add(", ".join(source_info))

        # 7. Tạo câu trả lời
        context = "\n".join(context_parts)
        prompt = create_prompt_combined(context, question, web_url)
        
        model_manager = ModelManager()
        model_manager.clean_gpu_memory()
        
        response = model_manager._pipe(prompt)[0]['generated_text']
        model_manager.clean_gpu_memory()
        
        # 8. Format câu trả lời cuối cùng
        answer = response.strip()
        if sources:
            answer += "\n\nNguồn tham khảo: " + "; ".join(sources)
        if web_url:
            answer += f"\n\nBạn có thể tham khảo thêm thông tin mới nhất tại: {web_url}"
            
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
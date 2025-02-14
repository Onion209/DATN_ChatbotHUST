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
            # Kiểm tra CUDA và clean memory
            if torch.cuda.is_available():
                self.clean_gpu_memory()
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA Version: {torch.version.cuda}")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adapter_path = os.path.join(current_dir, "models")
            
            # Load base model config
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            logger.info(f"Base model: {base_model_name}")
            
            # Load base model
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
            logger.info("Loaded tokenizer")
            
            # Load và merge LoRA adapter
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Tạo pipeline với max_length cao hơn
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
            
            logger.info("Created pipeline successfully")
            return pipe
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
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
    web_url = web_url if web_url else 'https://ts.hust.edu.vn/'
    
    template = """Bạn là một giáo viên phòng tư vấn tuyển sinh đại học Bách Khoa Hà Nội.
 Bạn có nhiệm vụ tư vấn tuyển sinh cho người dùng và trả lời đúng những câu hỏi mà người dùng đặt ra. Xưng hô với người dùng là em.
 Trả lời câu hỏi một người dùng thân thiện, chính xác. Nếu không tìm được thông tin hãy trả lời 'Vui lòng tra thêm thông tin trên website: {}'.

### Instruction:
{}

### Input:
{}

### Response:"""
    
    return template.format(web_url, question, context)

def get_data_from_db(question: str):
    """Lấy dữ liệu từ database và xử lý"""
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
            return None, web_url, None

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

        context = "\n".join(context_parts)
        return context, web_url, sources

    except Exception as e:
        logger.error(f"Error in get_data_from_db: {str(e)}")
        raise

def generate_answer(question: str, context: str, web_url: str, sources: set = None) -> str:
    """Sinh câu trả lời từ context và format kết quả"""
    try:
        # 1. Tạo prompt
        prompt = create_prompt_combined(context, question, web_url)
        
        # 2. Generate response
        try:
            model_manager = ModelManager()
            # Clean GPU memory trước khi generate
            model_manager.clean_gpu_memory()
            pipe = model_manager._pipe
            
            logger.info("Generating response...")
            response = pipe(prompt)[0]['generated_text']
            answer = response.strip()
            
            # Clean GPU memory sau khi generate
            model_manager.clean_gpu_memory()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
        
        # 3. Format câu trả lời cuối cùng
        if sources:
            answer += "\n\nNguồn tham khảo: " + "; ".join(sources)
        if web_url:
            answer += f"\n\nBạn có thể tham khảo thêm thông tin mới nhất tại: {web_url}"
            
        return answer

    except Exception as e:
        logger.error(f"Error in generate_answer: {str(e)}")
        raise

def test_combined(question: str) -> str:
    """Hàm chính kết hợp việc lấy data và sinh câu trả lời"""
    try:
        # 1. Lấy dữ liệu từ database
        context, web_url, sources = get_data_from_db(question)
        
        # 2. Xử lý trường hợp không có dữ liệu
        if context is None:
            fallback_message = (
                f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
            )
            return fallback_message
        
        # 3. Generate và trả về câu trả lời
        return generate_answer(question, context, web_url, sources)

    except Exception as e:
        logger.error(f"Error in test_combined: {str(e)}")
        return f"Lỗi: {str(e)}"

if __name__ == "__main__":

    question = "Em muốn tìm hiểu về ngành Kỹ thuật in thì liên hệ với ai ạ?"
    # context = "STT 01, Mã & Tên chương trình: (BF1) Kỹ thuật Sinh học, Họ và tên: PGS. Nguyễn Lan Hương, Email: huong.nguyenlan@hust.edu.vn, Điện thoại: 0903247172"
    # web_url = test_search(question)
    # source = None
    # generate_answer(question,context, web_url, source)
    print(get_data_from_db(question))
    # try:
    #     print("Initializing model...")
    #     model_manager = ModelManager()
    #     print("Model initialized successfully!")
        
    #     while True:
    #         question = input("\nHỏi (hoặc 'q' để thoát): ")
    #         if question.lower() == 'q':
    #             break
            
    #         result = test_combined(question)
    #         print(f"\nTrả lời: {result}")
            
    # except Exception as e:
    #     print(f"Error: {str(e)}")
    #     logger.error(f"Error in main loop: {str(e)}") 
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import logging
import gc  # Thêm import gc

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
        """Clean GPU memory"""
        if torch.cuda.is_available():
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            # Run garbage collector
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
            logger.info("Applied LoRA adapter")
            
            # Tạo pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=False
            )
            
            logger.info("Created pipeline successfully")
            return pipe
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise

def generate_response(prompt: str) -> str:
    try:
        model_manager = ModelManager()
        # Clean GPU memory trước khi generate
        model_manager.clean_gpu_memory()
        
        pipe = model_manager._pipe
        logger.info("Generating response...")
        response = pipe(prompt)[0]['generated_text']
        
        # Clean GPU memory sau khi generate
        model_manager.clean_gpu_memory()
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def test_model():
    test_prompt = """Bạn là một giáo viên phòng tư vấn tuyển sinh đại học Bách Khoa Hà Nội.
 Bạn có nhiệm vụ tư vấn tuyển sinh cho người dùng và trả lời đúng những câu hỏi mà người dùng đặt ra.
 Trả lời câu hỏi một người dùng thân thiện, chính xác. Nếu không tìm được thông tin hãy trả lời 'Vui lòng tra thêm thông tin trên website: 'https://ts.hust.edu.vn/'.

### Instruction:
{Trường mình là trường gì?}

### Input:
{Trường Đại học Bách khoa Hà Nội (tên tiếng Anh: Hanoi University of Science and Technology - HUST) 
    là trường đại học đầu ngành về kỹ thuật và công nghệ tại Việt Nam. 
    Địa chỉ: Số 1 Đại Cồ Việt, Hai Bà Trưng, Hà Nội.}

### Response:
{}"""
    response = generate_response(test_prompt)
    print("\nGenerated response:")
    print(response)
    return response

if __name__ == "__main__":
    print("Testing fine-tuned model...")
    result = test_model()
    print("\nTest completed successfully!")
        
#         # Test thêm câu hỏi khác
#         print("\nTesting another question...")
#         test_prompt2 = """Bạn là một giáo viên phòng tư vấn tuyển sinh đại học Bách Khoa Hà Nội.
#  Bạn có nhiệm vụ tư vấn tuyển sinh cho người dùng và trả lời đúng những câu hỏi mà người dùng đặt ra.
#  Trả lời câu hỏi một người dùng thân thiện, chính xác. Nếu không tìm được thông tin hãy trả lời 'Vui lòng tra thêm thông tin trên website: 'https://ts.hust.edu.vn/'.

# ### Instruction:
# {Trường có mấy chương trình đào tạo kỹ sư chất lượng cao?}

# ### Input:
# {Trường Đại học Bách khoa Hà Nội có 6 chương trình đào tạo kỹ sư chất lượng cao bằng tiếng Anh: 
#         ICT, ME, DSAI, HEDSPI, EPU, và TROY.}

# ### Response:
# {}"""
#         result2 = generate_response(test_prompt2)
#         print("\nGenerated response for second question:")
#         print(result2)
        
#     except Exception as e:
#         print(f"Test failed: {str(e)}") 
from main import (
    load_config, 
    load_model, 
    load_vector_db, 
    classify_question_domain,
    clear_gpu_memory
)
from model import (
    create_prompt,
    create_qa_chain,
    get_relevant_chunks,
    load_config,
    CustomRetriever
)
from search_web import test_search
from chatbot import generate_response, create_prompt_llama, get_data_with_context  
import logging
import os
import torch
import gc

logger = logging.getLogger(__name__)

class ChatBackend:
    def __init__(self):
        self.qa_chain = None
        self.current_model = None
        self.config = load_config()
        self.chat_history = []  # Lưu lịch sử trò chuyện
        
    @staticmethod
    def clean_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleaned GPU memory")

    def setup_qa_chain(self, embedding_model_choice="OpenAI", llm_model_choice="gpt-4o-mini"):
        try:
            api_key = self.config.get('OPENAI_API_KEY') if llm_model_choice == "gpt-4o-mini" else self.config.get('HUGGINGFACE_TOKEN')
            if not api_key:
                raise ValueError(f"API key not found for {llm_model_choice}")
                
            model = load_model(api_key, llm_model_choice)
            self.current_model = llm_model_choice
            return model
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise

    def get_domain_name(self, domain):
        domain_names = {
            "thong_tin_chung": "Thông Tin Chung",
            "de_an_tuyen_sinh": "Đề Án Tuyển Sinh",
            "xet_tuyen_tai_nang": "Xét Tuyển Tài Năng",
            "diem_chuan_tuyen_sinh": "Điểm Chuẩn Tuyển Sinh",
            "ky_thi_danh_gia_tu_duy": "Kỳ Thi Đánh Giá Tư Duy",
            "xac_thuc_chung_chi_ngoai_ngu": "Chứng Chỉ Ngoại Ngữ",
            "huong_nghiep": "Hướng Nghiệp",
            "Q_A": "Q&A Database"
        }
        return domain_names.get(domain, domain)

    def evaluate_chunk_with_gpt(self, chunk, query):
        """ Hàm này sử dụng GPT-4o-mini để đánh giá độ phù hợp của chunk với câu hỏi (true hoặc false). """
        try:
            model = self.setup_qa_chain(llm_model_choice="gpt-4o-mini")
            prompt = f"""
            Câu hỏi: {query}
            Đoạn văn: {chunk}
            Bạn hãy trả lời câu hỏi này với đoạn văn trên. Nếu đoạn văn trả lời đúng câu hỏi, trả lời 'true'. Nếu đoạn văn không trả lời đúng câu hỏi, trả lời 'false'.
            """
            result = model(prompt)
            return result.get("result", "").strip().lower() == "true"  # Kiểm tra xem kết quả có phải "true" không
        except Exception as e:
            logger.error(f"Error in evaluating chunk: {str(e)}")
            return False

    def evaluate_chunks(self, chunks, query):
        try:
            if not chunks:
                logger.warning("No chunks provided for evaluation.")
                return []

            evaluations = []
            for i, chunk in enumerate(chunks, 1):
                if not chunk:
                    logger.warning(f"Chunk {i} is empty.")
                    continue

                print(f"\nDEBUG - Câu hỏi: {query}\n")  # In câu hỏi
                print(f"DEBUG - Chunk {i}: {chunk}\n")  # In chunk để theo dõi
                
                # Đánh giá chunk với GPT-4o-mini
                is_relevant = self.evaluate_chunk_with_gpt(chunk, query)
                evaluation_result = "true" if is_relevant else "false"
                
                evaluations.append({
                    "chunk": chunk,
                    "evaluation": evaluation_result
                })
                print(f"Đánh giá Chunk {i}: {evaluation_result}\n")  # In ra kết quả đánh giá
            return evaluations

        except Exception as e:
            logger.error(f"Error evaluating chunks: {str(e)}")
            return [] 

    def get_chat_response(self, query, model_choice="gpt-4o-mini"):
        try:
            api_key = self.config.get('OPENAI_API_KEY') if model_choice == "gpt-4o-mini" else self.config.get('HUGGINGFACE_TOKEN')
            domain = classify_question_domain(query)
            domain_db, qa_db = load_vector_db(api_key, domain)
            web_url = test_search(query)

            # Tìm kiếm chunk từ cơ sở dữ liệu
            domain_docs = get_relevant_chunks(query, domain_db, None)
            qa_docs = get_relevant_chunks(query, None, qa_db)
            documents = domain_docs + qa_docs

            # Đánh giá từng chunk
            evaluations = self.evaluate_chunks(documents, query)
            
            # Câu trả lời dựa trên đánh giá
            any_true = any(eval['evaluation'] == 'true' for eval in evaluations)
            if not any_true:
                fallback_message = f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. Bạn có thể tham khảo thêm thông tin tại: {web_url}"
                return {"answer": fallback_message, "sources": [], "domain": domain}

            # Nếu có ít nhất 1 chunk đúng
            model = self.setup_qa_chain()
            prompt = create_prompt()
            qa_chain = create_qa_chain(domain_db, qa_db, model, prompt, web_url)
            result = qa_chain({"query": query})

            return {"answer": result["result"], "sources": [], "domain": domain}

        except Exception as e:
            logger.error(f"Error in get_chat_response: {str(e)}")
            return {"answer": "Đã xảy ra lỗi khi xử lý câu hỏi của bạn.", "sources": [], "domain": None}

def main():
    chat_backend = ChatBackend()
    query = input("Vui lòng nhập câu hỏi của bạn: ")  # Nhập câu hỏi từ người dùng
    response = chat_backend.get_chat_response(query)
    print(f"\nCâu trả lời: {response['answer']}")
    print(f"Nguồn: {response['sources']}")
    print(f"Miền: {response['domain']}")

if __name__ == "__main__":
    main()

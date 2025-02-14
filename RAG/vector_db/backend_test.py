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
# from test_combined import create_prompt_combined, test_combined, ModelManager, load_config as load_config_combined
from search_web import test_search
from chatbot import generate_response, create_prompt_llama, get_data_with_context  # Nhập các hàm từ chatbot.py
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
        self.chat_history = []  # Thêm biến lưu lịch sử chat
        
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

    def get_file_name(self, file_path):
        if not file_path or file_path == "Unknown":
            return "Unknown"
        # Lấy tên file từ đường dẫn đầy đủ
        base_name = os.path.basename(file_path)
        # Loại bỏ phần mở rộng file
        name_without_ext = os.path.splitext(base_name)[0]
        # Chuyển dấu gạch dưới thành khoảng trắng và viết hoa chữ đầu
        return name_without_ext.replace('_', ' ').title()

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

    def get_chat_response(self, query, model_choice=None):
        if model_choice == "gpt-4o-mini":
            try:
                api_key = self.config.get('OPENAI_API_KEY') if model_choice == "gpt-4o-mini" else self.config.get('HUGGINGFACE_TOKEN')
                
                domain = classify_question_domain(query)
                domain_db, qa_db = load_vector_db(api_key, domain)
                web_url = test_search(query)
                
                # Debug: In ra thông tin documents
                print("\nDEBUG - Documents from databases:")
                domain_docs = get_relevant_chunks(query, domain_db, None)
                qa_docs = get_relevant_chunks(query, None, qa_db)
                
                documents = domain_docs + qa_docs
                # Đánh giá các chunk
                evaluations = self.evaluate_chunks(documents, query)
                
                # Kiểm tra xem có chunk nào đúng không
                any_true = any(eval['evaluation'] == 'true' for eval in evaluations)  # Kiểm tra xem có ít nhất một chunk đúng
                
                if not any_true:
                    # Nếu không có chunk nào đúng, sinh URL cho người dùng
                    fallback_message = (
                        f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                        f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                        f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
                    )
                    return {
                        "answer": fallback_message,
                        "sources": [],
                        "domain": domain
                    }
                if not documents:
                    fallback_message = (
                        f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                        f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                        f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
                    )
                    return {
                        "answer": fallback_message,
                        "sources": [],
                        "domain": domain
                    }
                    
                model = self.setup_qa_chain(llm_model_choice=model_choice)
                prompt = create_prompt()
                qa_chain = create_qa_chain(domain_db, qa_db, model, prompt, web_url)
                
                result = qa_chain({"query": query})
                
                
                
                # Tổ chức sources theo domain
                sources = []
                
                # Xử lý documents từ qa_db
                if qa_docs:
                    qa_sources = []
                    for i, doc in enumerate(qa_docs, 1):
                        try:
                            content = doc.page_content.strip() if hasattr(doc, 'page_content') else ''
                            if len(content) > 200:
                                content = content[:200] + "..."
                            
                            score = 0.0
                            metadata = getattr(doc, 'metadata', {})
                            if isinstance(metadata, dict):
                                # Thử lấy score từ metadata nếu có
                                score = metadata.get('score', 0.0)
                                if not isinstance(score, (int, float)):
                                    # Nếu score không phải số, tính theo công thức cũ
                                    score = 0.8259 - (i-1)*0.0045
                            else:
                                # Nếu không có metadata, tính theo công thức cũ
                                score = 0.8259 - (i-1)*0.0045
                            
                            qa_sources.append({
                                "vector": f"Vector {i}",
                                "score": f"{float(score):.4f}",  # Đảm bảo score là string với 4 chữ số thập phân
                                "content": content
                            })
                        except Exception as e:
                            logger.error(f"Error processing qa document {i}: {str(e)}")
                            continue
                    
                    if qa_sources:
                        sources.append({
                            "name": "Q&A Database",
                            "vectors": qa_sources
                        })
                
                # Lấy câu trả lời từ result
                answer = result.get("result", "")  # Sử dụng get() để tránh KeyError
                if not answer and result.get("answer"):  # Thử lấy từ key "answer" nếu "result" không có
                    answer = result["answer"]
                
                response = {
                    "answer": answer,
                    "sources": sources,
                    "domain": domain
                }
                
                # Lưu vào lịch sử
                if not hasattr(self, 'chat_history'):
                    self.chat_history = []
                
                self.chat_history.append({
                    "question": query,
                    "response": response
                })
                
                return response
                
            except Exception as e:
                print(f"Error in get_chat_response: {str(e)}")
                logger.error(f"Error details: {str(e)}")
                return {
                    "answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.",
                    "sources": [],
                    "domain": domain
                }
        if model_choice == "llama":
            try:
                self.clean_gpu_memory()  # Dọn dẹp bộ nhớ GPU trước khi xử lý
                domain = classify_question_domain(query)
                logger.info(f"Classified domain: {domain}")
                
                # Khởi tạo domain_db và qa_db
                api_key = self.config.get('OPENAI_API_KEY')
                domain_db, qa_db = load_vector_db(api_key, domain)
                
                documents = get_relevant_chunks(query, domain_db, qa_db)
                context, web_url, sources = get_data_with_context(query)
                
                if not documents:
                    fallback_message = (
                        f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
                    )
                    return {
                        "answer": fallback_message,
                        "sources": [],
                        "domain": domain
                    }
                
                prompt = create_prompt_llama(query, context)  # Tạo prompt từ chatbot.py
                response = generate_response(prompt)  # Gọi hàm generate_response từ chatbot.py
                
                answer = response.strip()
                
                # Thêm nguồn tham khảo và URL
                if sources:
                    answer += "\n\nNguồn tham khảo: " + "; ".join(sources)
                if web_url:
                    answer += f"\n\nBạn có thể tham khảo thêm thông tin mới nhất tại: {web_url}"
                
                # Dọn dẹp bộ nhớ GPU
                self.clean_gpu_memory()
                return {
                    "answer": answer,
                    "sources": list(sources),
                    "domain": self.get_domain_name(domain)
                }

            except Exception as e:
                logger.error(f"Error in get_chat_response for Llama: {str(e)}")
                return {
                    "answer": f"Đã xảy ra lỗi: {str(e)}",
                    "sources": [],
                    "domain": None
                }
        
    def cleanup(self):
        try:
            clear_gpu_memory()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 

    def evaluate_chunks(self, chunks, query):
        try:
            if not chunks:  # Kiểm tra xem chunks có rỗng không
                logger.warning("No chunks provided for evaluation.")
                return []

            model = self.setup_qa_chain(llm_model_choice="gpt-4o-mini")
            prompt = '''Vai trò: Bạn là một chuyên gia xử lý ngôn ngữ tự nhiên. Nhiệm vụ của bạn là đánh giá mức độ phù hợp của một đoạn văn bản (chunk) với một câu hỏi nhất định.
            Yêu cầu đánh giá:
Dựa trên thông tin trong chunk, hãy đánh giá mức độ phù hợp của nó với câu hỏi theo các tiêu chí sau:

Độ liên quan (Relevance): Chunk có chứa thông tin cần thiết để trả lời câu hỏi không?

(0) Hoàn toàn không liên quan
(1) Ít liên quan, chỉ có một phần nhỏ thông tin phù hợp
(2) Có thông tin liên quan nhưng chưa đầy đủ
(3) Rất liên quan, có thể trả lời đầy đủ câu hỏi
Độ đầy đủ (Completeness): Chunk có đủ thông tin để trả lời câu hỏi không?

(0) Không cung cấp bất kỳ thông tin hữu ích nào
(1) Cung cấp một phần nhỏ nhưng không đủ để trả lời
(2) Cung cấp gần đủ thông tin nhưng vẫn thiếu một số phần quan trọng
(3) Cung cấp đầy đủ thông tin cần thiết
Tính chính xác (Accuracy): Thông tin trong chunk có chính xác không?

(0) Thông tin sai lệch hoặc không đúng ngữ cảnh
(1) Có một số lỗi nhỏ hoặc không rõ ràng
(2) Hầu hết thông tin là chính xác
(3) Hoàn toàn chính xác''' # Tạo prompt cho việc đánh giá
            evaluations = []

            for chunk in chunks:
                if not chunk:  # Kiểm tra xem chunk có rỗng không
                    logger.warning("Empty chunk encountered during evaluation.")
                    continue

                # Tạo prompt cho từng chunk
                evaluation_prompt = f"{prompt}\n\nCâu hỏi: {query}\n\nChunk: {chunk}\n\nĐánh giá xem chunk này có phù hợp với câu hỏi không."
                try:
                    result = model(evaluation_prompt)  # Gọi mô hình để đánh giá
                    evaluations.append({
                        "chunk": chunk,
                        "evaluation": result.get("result", "")
                    })
                except Exception as e:
                    logger.error(f"Error evaluating chunk: {chunk}. Error details: {str(e)}")
                    evaluations.append({
                        "chunk": chunk,
                        "evaluation": "Error during evaluation"
                    })

            return evaluations

        except Exception as e:
            logger.error(f"Error evaluating chunks: {str(e)}")
            return [] 
def main():
    # Khởi tạo đối tượng ChatBackend
    chat_backend = ChatBackend()
    config = load_config()
    # Câu hỏi mà bạn muốn kiểm tra
    query = "Em muốn tìm hiểu về chương trình Kỹ thuật Sinh học, em có thể liên hệ với ai ạ?"  # Thay thế bằng câu hỏi thực tế của bạn
    model_choice = "gpt-4o-mini"  # Hoặc "llama" nếu bạn muốn thử với mô hình khác
    api_key = config.get('OPENAI_API_KEY') if model_choice == "gpt-4o-mini" else self.config.get('HUGGINGFACE_TOKEN')

    domain = classify_question_domain(query)
    domain_db, qa_db = load_vector_db(api_key, domain)
    web_url = test_search(query)

# Debug: In ra thông tin documents
    print("\nDEBUG - Documents from databases:")
    domain_docs = get_relevant_chunks(query, domain_db, None)
    qa_docs = get_relevant_chunks(query, None, qa_db)
    
    documents = domain_docs + qa_docs
    print(documents)
    # output = ChatBackend.evaluate_chunks(ChatBackend, documents, query)
    # print(output)
                # Đánh giá các chunk
    # Gọi hàm get_chat_response
    # response = chat_backend.get_chat_response(query, model_choice)

    # # In ra câu trả lời
    # print("Câu trả lời:", response['answer'])
    # print("Nguồn:", response['sources'])
    # print("Miền:", response['domain'])

if __name__ == "__main__":
    main()
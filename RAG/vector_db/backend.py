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
    CustomRetriever
)
from test_combined import create_prompt_combined, test_combined, ModelManager, load_config as load_config_combined
from search_web import test_search
import logging
import os

logger = logging.getLogger(__name__)

class ChatBackend:
    def __init__(self):
        self.qa_chain = None
        self.current_model = None
        self.config = load_config()
        self.chat_history = []  # Thêm biến lưu lịch sử chat
        
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
                api_key = self.config.get('OPENAI_API_KEY') 
                # Log để debug
                logger.info(f"Processing query with model: {model_choice}")
                
                # Phân loại domain và lấy dữ liệu
                domain = classify_question_domain(query)
                logger.info(f"Classified domain: {domain}")
                
                domain_db, qa_db = load_vector_db(api_key, domain)
                web_url = test_search(query)
                
                # Lấy documents từ vector DB
                documents = get_relevant_chunks(query, domain_db, qa_db)
                logger.info(f"Retrieved {len(documents)} documents")
                
                # Xử lý documents trước khi tạo context
                context_parts = []
                sources = set()
                
                for doc in documents:
                    score = float(doc.metadata.get('score', 0))
                    if score >= 0.9:
                        context_parts.append(doc.page_content)
                        metadata = doc.metadata
                        source_info = []
                        
                        source = metadata.get('source', 'Unknown')
                        page = metadata.get('page', 'Unknown')
                        source_name = self.get_file_name(source)
                        
                        if source_name != "Unknown":
                            if page != "Unknown":
                                source_info.append(f"{source_name} (trang {page})")
                            else:
                                source_info.append(source_name)
                        
                        if source_info:
                            sources.add(", ".join(source_info))

                if not documents:
                    fallback_message = (
                        f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                        f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                        f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
                    )
                    return {
                        "answer": fallback_message,
                        "sources": [],
                        "domain": self.get_domain_name(domain)
                    }

                context = "\n".join(context_parts)
                logger.info(f"Created context with length: {len(context)}")
                model_manager = ModelManager()
                prompt = create_prompt_combined(context, query, web_url)
                response = model_manager._pipe(prompt)[0]['generated_text']
                answer = response.strip()
                
                # Thêm nguồn tham khảo và URL
                if sources:
                    answer += "\n\nNguồn tham khảo: " + "; ".join(sources)
                if web_url:
                    answer += f"\n\nBạn có thể tham khảo thêm thông tin mới nhất tại: {web_url}"
                
                # Dọn dẹp bộ nhớ GPU
                model_manager.clean_gpu_memory()
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
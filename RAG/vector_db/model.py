from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from transformers import pipeline
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
import requests
from search_web import test_search
# Cấu hình logging chỉ cho __main__
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        # Sử dụng os.path.normpath để chuẩn hóa đường dẫn theo hệ điều hành
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
            login(token=api_key)
            
            pipe = pipeline(
                "text-generation",
                model="meta-llama/Llama-2-7b-chat-hf",
                token=api_key,
                temperature=0.5,
                max_length=2048,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

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

def classify_question_domain(question):
    """
    Phân loại câu hỏi thuộc miền nào
    Returns: tên collection trong ChromaDB
    """
    # Định nghĩa từ khóa cho từng miền
    domains = {
        "thong_tin_chung": ["thông tin chung", "giới thiệu", "tổng quan", "trường", "bách khoa"],
        "de_an_tuyen_sinh": ["đề án", "chỉ tiêu", "phương thức"],
        "xet_tuyen_tai_nang": ["tài năng", "xét tuyển tài năng", "năng khiếu"],
        "diem_chuan_tuyen_sinh": [
            "điểm chuẩn", "điểm trúng tuyển", "điểm đầu vào", 
            "điểm thi", "điểm xét tuyển",
            "điểm chuẩn 2024", "điểm trúng tuyển 2024",  # Thêm từ khóa kết hợp
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
    
    return "Q_A"  # Collection mặc định nếu không phân loại được

def load_vector_db(api_key, collection_name):
    try:
        config = load_config()
        # Sử dụng đường dẫn tương đối từ thư mục hiện tại
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persist_directory = os.path.join(current_dir, "chroma_db")
        
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        # Tạo thư mục collection nếu chưa tồn tại
        collection_path = os.path.join(persist_directory, collection_name)
        if not os.path.exists(collection_path):
            os.makedirs(collection_path)
            
        # Tạo thư mục Q_A nếu chưa tồn tại 
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

def get_relevant_chunks(question, domain_db, qa_db, k=3):
    try:
        # Thử tìm trong domain_db
        try:
            domain_results = domain_db.similarity_search_with_relevance_scores(question, k=k)
        except:
            domain_results = []
        
        # Thử tìm trong qa_db
        try:
            qa_results = qa_db.similarity_search_with_relevance_scores(question, k=k)
        except:
            qa_results = []
            
        # Log the size of results
        logger.debug(f"Domain results count: {len(domain_results)}")
        logger.debug(f"QA results count: {len(qa_results)}")
        
        # Chọn kết quả phù hợp nhất
        results = []
        if domain_results:  # Nếu có kết quả từ domain_db
            results = domain_results
        if not results and qa_results:  # Nếu không có kết quả từ domain_db và có kết quả từ qa_db
            results = qa_results
            
        if not results:  # Nếu không có kết quả từ cả hai db
            return []

        # Chuyển đổi kết quả sang format Document
        documents = []
        for doc, score in results:
            # Giữ lại tất cả metadata từ document gốc
            metadata = doc.metadata.copy()  # Copy metadata gốc
            metadata.update({
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'Unknown'),
                'score': f"{score:.4f}",
                'doc_id': doc.metadata.get('doc_id', None)  # Thêm doc_id
            })
            
            documents.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))
        
        return documents
        
    except Exception as e:
        logger.error(f"Error getting relevant chunks: {str(e)}")
        return []

def process_chunks(docs, all_docs, selected_chunks):
    """
    Helper function để xử lý chunks từ một database
    """
    try:
        doc_ids = all_docs.get('ids', [])
        
        for doc in docs:
            doc_id = doc.metadata.get('doc_id')
            if not doc_id:
                continue
                
            try:
                current_index = doc_ids.index(doc_id)
                selected_chunks.add(current_index)
                
                # Thêm chunk trước
                if current_index > 0:
                    selected_chunks.add(current_index - 1)
                    
                # Thêm chunk sau
                if current_index < len(doc_ids) - 1:
                    selected_chunks.add(current_index + 1)
            except ValueError:
                continue
    except Exception as e:
        logger.warning(f"Error processing chu1  nks: {str(e)}")
        # Không raise exception để tránh làm gián đoạn luồng chính

class CustomRetriever(BaseRetriever, BaseModel):
    domain_db: Any = Field(default=None)
    qa_db: Any = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, domain_db, qa_db):
        super().__init__()
        self.domain_db = domain_db
        self.qa_db = qa_db

    def get_relevant_documents(self, query: str) -> List[Document]:
        return get_relevant_chunks(query, self.domain_db, self.qa_db)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")

def create_qa_chain(domain_db, qa_db, model, prompt, web_url):
    try:
        retriever = CustomRetriever(domain_db, qa_db)
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt.partial(web_url=web_url)
            }
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        raise


def get_answer(question, model_type="gpt-4o-mini"):
    try:
        config = load_config()
        api_key = config.get('OPENAI_API_KEY') if model_type == "gpt-4o-mini" else config.get('HUGGINGFACE_TOKEN')
        
        if not api_key:
            raise Exception(f"API key not found for {model_type} in config.json")
        
        domain = classify_question_domain(question)
        model = load_model(api_key, model_type)
        domain_db, qa_db = load_vector_db(api_key, domain)
        web_url = test_search(question)
        
        # Lấy documents và kiểm tra
        documents = get_relevant_chunks(question, domain_db, qa_db)
        if not documents:
            fallback_message = (
                f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
            )
            return {
                "answer": fallback_message,
                "source_documents": [],
                "domain": domain
            }
            
        prompt = create_prompt()
        qa_chain = create_qa_chain(domain_db, qa_db, model, prompt, web_url)
        
        # Thêm xử lý kết quả
        result = qa_chain({"query": question})
        answer = result["result"]
        
        # In ra câu trả lời
        print("\nCâu trả lời:")
        print(answer)
        print("\nNguồn tham khảo:")
        for doc in result.get("source_documents", []):
            print(f"- {doc.metadata.get('source', 'Unknown')}")
        
        return {
            "answer": answer,
            "source_documents": result.get("source_documents", []),
            "domain": domain
        }
        
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        raise

def get_db_path():
    """Trả về đường dẫn chuẩn đến thư mục chroma_db"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "chroma_db")
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    return os.path.normpath(db_path)

if __name__ == "__main__":
    print("\nChọn mô hình:")
    print("1. GPT-4o-mini")
    print("2. Llama-2-7b-chat")
    
    model_choice = input("\nChọn model (1-2): ")
    model_type = "gpt-4o-mini" if model_choice == "1" else "llama"
    
    while True:
        question = input("\nCâu hỏi: ")
        if question.lower() in ['quit', 'q', 'exit']:
            break
            
        try:
            result = get_answer(question, model_type)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Error in main loop: {str(e)}")
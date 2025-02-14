from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field
from typing import List, Any
import logging
import os

logger = logging.getLogger(__name__)

def classify_question_domain(question):
    """
    Phân loại câu hỏi thuộc miền nào
    Returns: tên collection trong ChromaDB
    """
    domains = {
        "thong_tin_chung": ["thông tin chung", "giới thiệu", "tổng quan", "trường", "bách khoa"],
        "de_an_tuyen_sinh": ["đề án", "chỉ tiêu", "phương thức"],
        "xet_tuyen_tai_nang": ["tài năng", "xét tuyển tài năng", "năng khiếu"],
        "diem_chuan_tuyen_sinh": [
            "điểm chuẩn", "điểm trúng tuyển", "điểm đầu vào", 
            "điểm thi", "điểm xét tuyển",
            "điểm chuẩn 2024", "điểm trúng tuyển 2024",
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
    
    return "Q_A"

def get_relevant_chunks(question, domain_db, qa_db, k=3):
    try:
        try:
            domain_results = domain_db.similarity_search_with_relevance_scores(question, k=k)
        except:
            domain_results = []
        
        try:
            qa_results = qa_db.similarity_search_with_relevance_scores(question, k=k)
        except:
            qa_results = []
            
        logger.debug(f"Domain results count: {len(domain_results)}")
        logger.debug(f"QA results count: {len(qa_results)}")
        
        results = []
        if domain_results:
            results = domain_results
        if not results and qa_results:
            results = qa_results
            
        if not results:
            return []

        documents = []
        for doc, score in results:
            metadata = doc.metadata.copy()
            metadata.update({
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'Unknown'),
                'score': f"{score:.4f}",
                'doc_id': doc.metadata.get('doc_id', None)
            })
            
            documents.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))
        
        return documents
        
    except Exception as e:
        logger.error(f"Error getting relevant chunks: {str(e)}")
        return []

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

def get_db_path():
    """Trả về đường dẫn chuẩn đến thư mục chroma_db"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "chroma_db")
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    return os.path.normpath(db_path)

def test_retrieval():
    """
    Hàm test thử việc trích xuất chunks từ câu hỏi
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
        import json
        
        # Load config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            
        api_key = config.get('OPENAI_API_KEY')
        if not api_key:
            raise Exception("Không tìm thấy OPENAI_API_KEY trong config.json")

        print("\nKhởi tạo embedding model và vector stores...")
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        
        while True:
            question = input("\nNhập câu hỏi của bạn (nhập 'q' để thoát): ")
            if question.lower() in ['q', 'quit', 'exit']:
                break
                
            # Xác định domain
            domain = classify_question_domain(question)
            print(f"\nDomain được phân loại: {domain}")
            
            # Khởi tạo vector stores
            domain_db = Chroma(
                persist_directory=os.path.join(get_db_path(), domain),
                embedding_function=embedding_model
            )
            
            qa_db = Chroma(
                persist_directory=os.path.join(get_db_path(), "Q_A"),
                embedding_function=embedding_model
            )
            
            # Lấy chunks liên quan
            chunks = get_relevant_chunks(question, domain_db, qa_db)
            
            print(f"\nTìm thấy {len(chunks)} chunks liên quan:")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n--- Chunk {i} ---")
                print(f"Nội dung: {chunk.page_content}")
                print(f"Metadata: ")
                for key, value in chunk.metadata.items():
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"Lỗi khi test retrieval: {str(e)}")

if __name__ == "__main__":
    print("Chương trình test retrieval chunks")
    test_retrieval() 
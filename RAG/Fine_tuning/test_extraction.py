from langchain_openai import OpenAIEmbeddings
import os
import json
import logging
import sys
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
        
        # 3. Load vector databases và kiểm tra
        domain_db, qa_db = load_vector_db(api_key, domain)
        print("Đã load vector databases")
        
        # Thêm debug info cho database
        domain_docs = domain_db.get()
        qa_docs = qa_db.get()
        print(f"Domain DB size: {len(domain_docs['ids']) if domain_docs else 0} documents")
        print(f"QA DB size: {len(qa_docs['ids']) if qa_docs else 0} documents")
        
        # 4. Lấy chunks với relevance scores
        documents = get_relevant_chunks(question, domain_db, qa_db, k=3)
        
        # Debug info cho chunks
        print(f"Số lượng chunks tìm được: {len(documents)}")
        
        # 5. Xử lý trường hợp không có chunks
        if not documents:
            print("\nKhông tìm thấy thông tin liên quan.")
            return None, web_url, None

        # 6. Xử lý và sắp xếp chunks theo score
        documents.sort(key=lambda x: float(x.metadata.get('score', 0)), reverse=True)
        
        # 7. Tạo context từ documents
        context_parts = []
        sources = set()
        
        print("\nCác chunks tìm được:")
        print("=" * 50)
        
        for doc in documents:
            print(f"\nNội dung chunk:")
            print(f"{doc.page_content}")
            print(f"Độ tin cậy: {doc.metadata.get('score', 'Unknown')}")
            
            context_parts.append(doc.page_content)
            metadata = doc.metadata
            
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"- {key}: {value}")
            
            source = metadata.get('source')
            page = metadata.get('page')
            
            if source and source != 'Unknown':
                source_info = source
                if page and page != 'Unknown':
                    source_info += f" (trang {page})"
                sources.add(source_info)
                
            print("=" * 50)

        context = "\n".join(context_parts)
        
        # In thông tin tổng hợp
        print(f"\nTổng số chunks phù hợp: {len(context_parts)}")
        
        if sources:
            print("\nNguồn tham khảo:")
            for source in sources:
                print(f"- {source}")

        if context:
            print("\nContext tổng hợp:")
            print("-" * 50)
            print(context)
            print("-" * 50)
            
        return context, web_url, sources

    except Exception as e:
        logger.error(f"Error in get_data_from_db: {str(e)}")
        print(f"Lỗi: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    try:
        while True:
            question = input("\nNhập câu hỏi (hoặc 'q' để thoát): ")
            if question.lower() == 'q':
                break
            
            context, web_url, sources = get_data_from_db(question)
            if context is None:
                print(f"\nKhông tìm thấy thông tin phù hợp. Bạn có thể tham khảo thêm tại: {web_url}")
                
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        logger.error(f"Error in main: {str(e)}") 
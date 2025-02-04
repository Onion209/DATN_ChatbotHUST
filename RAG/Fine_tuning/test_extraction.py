from langchain_openai import OpenAIEmbeddings
import os
import json
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_db.model import (
    classify_question_domain,
    load_vector_db,
    get_relevant_chunks
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(current_dir, "vector_db", "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def test_extraction(question: str):
    try:
        # Load config
        config = load_config()
        api_key = config.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found in config")

        # Phân loại domain
        domain = classify_question_domain(question)
        print(f"\nDomain được phân loại: {domain}")
        
        # Load vector databases
        domain_db, qa_db = load_vector_db(api_key, domain)
        print("Đã load vector databases")
        
        # Get relevant chunks
        chunks = get_relevant_chunks(question, domain_db, qa_db)
        if not chunks:
            print("\nKhông tìm thấy thông tin liên quan.")
            return

        # In chi tiết từng chunk
        print("\nCác chunks tìm được:")
        print("=" * 50)
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(f"Nội dung: {chunk.page_content}")
            print("\nMetadata:")
            for key, value in chunk.metadata.items():
                print(f"- {key}: {value}")
            print("=" * 50)

        # Tổng hợp sources
        sources = set()
        for chunk in chunks:
            source = chunk.metadata.get('source')
            page = chunk.metadata.get('page')
            if source and source != 'Unknown':
                if page and page != 'Unknown':
                    sources.add(f"{source} (trang {page})")
                else:
                    sources.add(source)

        if sources:
            print("\nNguồn tham khảo:")
            for source in sources:
                print(f"- {source}")

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        logger.error(f"Error in test_extraction: {str(e)}")

if __name__ == "__main__":
    while True:
        question = input("\nNhập câu hỏi (hoặc 'q' để thoát): ")
        if question.lower() == 'q':
            break
        
        test_extraction(question) 
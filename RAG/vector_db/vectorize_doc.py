import os
import json
import logging
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import csv
from openai import OpenAI
from langchain.schema import Document

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    try:
        # Lấy đường dẫn tuyệt đối của thư mục hiện tại
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        logger.error(f"config.json file not found at {config_path}")
        raise Exception(f"config.json file not found at {config_path}")
    except json.JSONDecodeError:
        logger.error("Error decoding config.json")
        raise Exception("Error decoding config.json")

def load_documents_from_folder(folder_path):
    if not os.path.exists(folder_path):
        logger.error(f"Folder path does not exist: {folder_path}")
        raise Exception(f"Folder path does not exist: {folder_path}")
        
    documents_by_category = {}
    
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                category = os.path.relpath(root, folder_path)  # Get the subdirectory name
                if category not in documents_by_category:
                    documents_by_category[category] = []
                
                try:
                    if file.endswith('.pdf'):
                        loader_pdf = PyPDFLoader(file_path)
                        documents_by_category[category].extend(loader_pdf.load())
                        logger.info(f"Loaded PDF: {file_path}")
                    elif file.endswith('.docx'):
                        loader_doc = Docx2txtLoader(file_path)
                        documents_by_category[category].extend(loader_doc.load())
                        logger.info(f"Loaded DOCX: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {str(e)}")
                    continue
    
        logger.info(f"Total categories loaded: {len(documents_by_category)}")
        return documents_by_category
    except Exception as e:
        logger.error(f"Error in load_documents_from_folder: {str(e)}")
        raise

def process_csv_and_store_embeddings(csv_file_path, text_splitter, persist_directory, api_key):
    # Khởi tạo OpenAIEmbeddings từ LangChain
    embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
    
    combined_text = ""
    
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            context = row['context']
            question = row['question']
            answer = row['answer']
            
            # Kết hợp context, question, và answer thành một chuỗi văn bản
            text_block = f"{context} {question} {answer}"
            
            # Gộp tất cả các text block thành một văn bản lớn
            combined_text += text_block + " "  # Thêm khoảng trắng giữa các text block
    
    # Chia văn bản lớn thành các chunk
    chunks = text_splitter.split_text(combined_text)
    
    # Tạo documents từ chunks
    documents = [Document(page_content=chunk) for chunk in chunks]
    logger.info(f"Created {len(documents)} documents from CSV data")
    
    # Lưu vào ChromaDB và để Chroma tự động xử lý embedding
    category_persist_directory = os.path.join(persist_directory, "Q_A")
    if not os.path.exists(category_persist_directory):
        os.makedirs(category_persist_directory)
        os.chmod(category_persist_directory, 0o777)
    
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=category_persist_directory
    )
    
    # Log thông tin về các vector đã được embedding
    collection = db.get()
    logger.info(f"Successfully embedded and stored {len(collection['ids'])} vectors for Q&A")
    logger.info(f"First few document contents: {[doc[:100] + '...' for doc in collection['documents'][:3]]}")

def create_db(documents_by_category, api_key):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=50
        )
        
        # Chỉ định đường dẫn tới thư mục có quyền ghi
        persist_directory = os.path.join(os.path.expanduser("~"), "DATN_test", "chroma_db")
        
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            # Set quyền 777 cho thư mục
            os.chmod(persist_directory, 0o777)
            logger.info(f"Created directory: {persist_directory}")

        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        
        for category, documents in documents_by_category.items():
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from documents in category: {category}")

            # Khởi tạo Chroma với client_settings cho từng category
            category_persist_directory = os.path.join(persist_directory, category)
            if not os.path.exists(category_persist_directory):
                os.makedirs(category_persist_directory)
                os.chmod(category_persist_directory, 0o777)
                logger.info(f"Created directory for category: {category_persist_directory}")

            db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=category_persist_directory
            )
            
            # Log thông tin về các vector đã được embedding
            collection = db.get()
            logger.info(f"Successfully embedded and stored {len(collection['ids'])} vectors for category: {category}")
            logger.info(f"Sample documents from {category}: {[doc[:100] + '...' for doc in collection['documents'][:3]]}")
        
        logger.info("Database created successfully for all categories")
        return db
    except Exception as e:
        logger.error(f"Error in create_db: {str(e)}")
        raise

def main():
    try:
        # Load config và lấy API key
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY not found in config.json")
            raise Exception("OPENAI_API_KEY not found in config.json")

        file_path = "/home/minhlahanhne/DATN_test/RAG/data"
        documents = load_documents_from_folder(file_path)
        db = create_db(documents, api_key)
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=50
        )
        
        # Process CSV file
        csv_file_path = "/home/minhlahanhne/DATN_test/RAG/vector_db/Q&A&C.csv"
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        persist_directory = os.path.join(os.path.expanduser("~"), "DATN_test", "chroma_db")
        process_csv_and_store_embeddings(csv_file_path, text_splitter, persist_directory, api_key)
        
        logger.info("Documents and Q&A vectorized and saved to ChromaDB successfully")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
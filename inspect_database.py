import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load config từ file config.json"""
    try:
        config_path = os.path.join("config.json")
        with open(config_path, 'r') as config_file:
            return json.load(config_file)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def inspect_collection(api_key, collection_name=None):
    """
    Kiểm tra nội dung của một collection cụ thể hoặc liệt kê tất cả collections
    """
    try:
        persist_directory = os.path.join(os.path.expanduser("~"), "DATN_test", "chroma_db")
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Liệt kê tất cả collections có sẵn
        available_collections = [d for d in os.listdir(persist_directory) 
                               if os.path.isdir(os.path.join(persist_directory, d))]
        print("\nAvailable collections:")
        for i, coll in enumerate(available_collections, 1):
            print(f"{i}. {coll}")

        if collection_name is None:
            # Cho phép user chọn collection
            choice = input("\nChọn số thứ tự collection muốn xem (hoặc 'all' để xem tất cả): ")
            if choice.lower() == 'all':
                for coll in available_collections:
                    print(f"\n{'='*80}")
                    print(f"Examining collection: {coll}")
                    print(f"{'='*80}")
                    inspect_single_collection(persist_directory, coll, embedding_model)
                return
            else:
                try:
                    idx = int(choice) - 1
                    collection_name = available_collections[idx]
                except (ValueError, IndexError):
                    print("Lựa chọn không hợp lệ!")
                    return
        
        # Kiểm tra một collection cụ thể
        inspect_single_collection(persist_directory, collection_name, embedding_model)
            
    except Exception as e:
        logger.error(f"Error during inspection: {str(e)}")
        raise

def inspect_single_collection(persist_directory, collection_name, embedding_model):
    """Kiểm tra chi tiết một collection"""
    try:
        db = Chroma(
            persist_directory=os.path.join(persist_directory, collection_name),
            embedding_function=embedding_model
        )
        
        results = db.get()
        
        print(f"\nCollection: {collection_name}")
        print(f"Số lượng documents: {len(results['documents'])}")
        
        for i, doc in enumerate(results['documents'], 1):
            print(f"\n{'='*40}")
            print(f"Document {i}:")
            print(f"{'='*40}")
            print(doc)
            
            if 'metadatas' in results and results['metadatas']:
                print(f"\nMetadata:")
                print(results['metadatas'][i-1])
            print(f"{'='*40}")
            
            # Cho phép user kiểm soát việc xem từng document
            if i < len(results['documents']):
                user_input = input("\nNhấn Enter để xem document tiếp theo (hoặc 'q' để thoát): ")
                if user_input.lower() == 'q':
                    break
                
    except Exception as e:
        logger.error(f"Error inspecting collection {collection_name}: {str(e)}")
        print(f"Error examining collection {collection_name}: {str(e)}")

def main():
    try:
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found in config")
        
        while True:
            inspect_collection(api_key)
            
            continue_inspection = input("\nBạn có muốn kiểm tra collection khác không? (y/n): ")
            if continue_inspection.lower() != 'y':
                break
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
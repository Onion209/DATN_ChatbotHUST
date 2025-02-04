import pandas as pd
from model import get_answer
import json
import time
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model():
    try:
        # Đọc file câu hỏi
        questions_df = pd.read_csv('/home/minhlahanhne/DATN_test/RAG/vector_db/questions_only.csv')
        
        # Khởi tạo list để lưu kết quả
        results = []
        csv_results = []  # List riêng cho kết quả CSV
        
        # Chạy từng câu hỏi với progress bar
        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing questions"):
            question = row['question']
            if not isinstance(question, str) or len(question.strip()) == 0:
                continue
                
            try:
                # Thêm delay nhỏ để tránh rate limit
                time.sleep(0.5)
                
                # Lấy câu trả lời từ model
                response = get_answer(question, model_type="gpt4")
                
                # Lưu kết quả đầy đủ cho JSON
                results.append({
                    'question': question,
                    'answer': response['answer'],
                    'domain': response['domain'],
                    'sources': [
                        {
                            'content': doc.page_content,
                            'source': doc.metadata.get('source', ''),
                            'score': doc.metadata.get('score', '')
                        }
                        for doc in response.get('source_documents', [])
                    ]
                })
                
                # Lưu kết quả đơn giản cho CSV
                csv_results.append({
                    'question': question,
                    'answer': response['answer']
                })
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                error_result = {
                    'question': question,
                    'answer': f"Error: {str(e)}"
                }
                csv_results.append(error_result)
                results.append({**error_result, 'error': str(e)})
                
        # Lưu kết quả chi tiết ra file JSON
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # Lưu kết quả đơn giản ra file CSV
        pd.DataFrame(csv_results).to_csv('evaluation_results.csv', index=False, encoding='utf-8')
        
        logger.info("Evaluation complete. Results saved to evaluation_results.json and evaluation_results.csv")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model() 
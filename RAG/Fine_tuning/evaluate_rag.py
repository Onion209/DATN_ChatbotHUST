import pandas as pd
import time
import json
from tqdm import tqdm
import torch
import gc
import os
import sys
import importlib
from RAG_finetune import get_answer

def force_cuda_empty():
    """Giải phóng bộ nhớ GPU một cách mạnh mẽ"""
    if torch.cuda.is_available():
        # Giải phóng bộ nhớ CUDA
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Xóa tất cả tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except Exception as e:
                pass
        
        # Gọi garbage collector
        gc.collect()
        
        # Reset CUDA
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def evaluate_all_questions():
    try:
        questions_df = pd.read_csv('Fine_tuning/questions_only.csv')
        results = []
        
        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing questions"):
            question = row['question']
            print("\nCâu hỏi:", question)
            
            try:
                # Giải phóng bộ nhớ GPU
                force_cuda_empty()
                
                # Đợi GPU reset
                time.sleep(2)
                
                # Reload module để đảm bảo model được khởi tạo mới
                importlib.reload(sys.modules['RAG_finetune'])
                from RAG_finetune import get_answer
                
                # Xử lý câu hỏi
                answer = get_answer(question)
                
                results.append({
                    'question': question,
                    'answer': answer
                })
                
                print("-" * 100)
                
                # Giải phóng bộ nhớ sau khi xử lý
                force_cuda_empty()
                
            except Exception as e:
                print(f"Error processing question: {question}")
                print(f"Error details: {str(e)}")
                results.append({
                    'question': question,
                    'answer': None
                })
            
            # Lưu kết quả tạm thời
            temp_df = pd.DataFrame(results)
            temp_df.to_csv('Fine_tuning/evaluation_results_temp.csv', index=False, encoding='utf-8')
        
        # Lưu kết quả cuối cùng
        results_df = pd.DataFrame(results)
        results_df.to_csv('Fine_tuning/evaluation_results.csv', index=False, encoding='utf-8')
        
        print(f"\nEvaluation complete. Processed {len(results)} questions.")
        print("Results saved to evaluation_results.csv")
        
    except Exception as e:
        print(f"Fatal error during evaluation: {str(e)}")

if __name__ == "__main__":
    evaluate_all_questions() 
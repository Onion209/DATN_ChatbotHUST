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
from search_web import test_search
from chatbot import generate_response, create_prompt_llama, get_data_with_context
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
        self.chat_history = []  # Lưu lịch sử chat
        
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
(3) Hoàn toàn chính xác'''

            evaluations = []
            for i, chunk in enumerate(chunks):
                # Lấy chunk trước (i-1), chunk hiện tại (i) và chunk sau (i+1)
                prev_chunk = chunks[i-1] if i > 0 else None
                next_chunk = chunks[i+1] if i < len(chunks) - 1 else None
                combined_chunk = f"{prev_chunk}\n{chunk}\n{next_chunk}" if prev_chunk and next_chunk else chunk

                # Tạo prompt cho từng chunk
                evaluation_prompt = f"{prompt}\n\nCâu hỏi: {query}\n\nChunk: {combined_chunk}\n\nĐánh giá xem chunk này có phù hợp với câu hỏi không."
                try:
                    result = model(evaluation_prompt)  # Gọi mô hình để đánh giá
                    evaluation = result.get("result", "").lower()
                    evaluations.append({
                        "chunk": chunk,
                        "evaluation": evaluation
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

    def get_chat_response(self, query, model_choice=None):
        if model_choice == "gpt-4o-mini":
            try:
                api_key = self.config.get('OPENAI_API_KEY') if model_choice == "gpt-4o-mini" else self.config.get('HUGGINGFACE_TOKEN')
                domain = classify_question_domain(query)
                domain_db, qa_db = load_vector_db(api_key, domain)
                web_url = test_search(query)

                domain_docs = get_relevant_chunks(query, domain_db, None)
                qa_docs = get_relevant_chunks(query, None, qa_db)
                
                documents = domain_docs + qa_docs
                evaluations = self.evaluate_chunks(documents, query)
                
                # Kiểm tra xem có chunk nào được đánh giá là 'true'
                true_chunks = [eval for eval in evaluations if eval['evaluation'] == 'true']
                
                if not true_chunks:
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
                
                answer = result.get("result", "") or result.get("answer", "")
                
                response = {
                    "answer": answer,
                    "sources": [],
                    "domain": domain
                }
                
                self.chat_history.append({
                    "question": query,
                    "response": response
                })
                
                return response
            except Exception as e:
                logger.error(f"Error in get_chat_response: {str(e)}")
                return {
                    "answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.",
                    "sources": [],
                    "domain": None
                }

def main():
    chat_backend = ChatBackend()
    query = "Em muốn tìm hiểu về chương trình Kỹ thuật Sinh học, em có thể liên hệ với ai ạ?"
    model_choice = "gpt-4o-mini"
    response = chat_backend.get_chat_response(query, model_choice)
    print(response['answer'])

if __name__ == "__main__":
    main()

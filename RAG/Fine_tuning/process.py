import pandas as pd

# Đọc file CSV
df = pd.read_csv('Fine_tuning/evaluation_results_temp.csv')

# Hàm để làm sạch câu trả lời
def clean_answer(answer):
    # Nếu có "Answer:" trong câu trả lời, lấy phần sau "Answer:"
    if "Answer:" in answer:
        answer = answer.split("Answer:")[1].strip()
    
    # Loại bỏ các phần text không cần thiết
    answer = answer.split("Question:")[0].strip()
    answer = answer.split("Trả lời câu hỏi")[0].strip()
    
    # Loại bỏ các dòng trống và khoảng trắng thừa
    answer = ' '.join(line.strip() for line in answer.splitlines() if line.strip())
    
    return answer

# Áp dụng hàm clean_answer cho cột answer
df['answer'] = df['answer'].apply(clean_answer)

# Lưu lại file CSV đã được làm sạch
df.to_csv('Fine_tuning/evaluation_results_clean.csv', index=False)
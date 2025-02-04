import pandas as pd

# Đọc file CSV gốc
df = pd.read_csv('RAG/vector_db/Q&A&C - Test_model.csv')

# Tạo DataFrame mới chỉ với cột question
questions_df = pd.DataFrame(df['question'])

# Lưu vào file CSV mới
questions_df.to_csv('RAG/vector_db/questions_only.csv', index=False)
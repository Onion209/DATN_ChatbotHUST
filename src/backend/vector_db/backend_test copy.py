# Thay đổi phần in ra thông tin documents
print("\nDEBUG - Documents from databases:")
for doc in documents:
    print(doc.page_content)  # Chỉ in ra nội dung của document

# Thay đổi phần hiển thị đánh giá
for i, eval in enumerate(evaluations):
    print(f"Đánh giá cho chunk {i + 1}: {eval['evaluation']}\n")  # In ra đánh giá

# Thay đổi phần in ra câu trả lời
print(f"Câu trả lời của LLM: {answer}\n")  # In ra câu trả lời 
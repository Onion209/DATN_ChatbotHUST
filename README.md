# 🎓 Trợ Lý Ảo Tuyển Sinh - Đại học Bách Khoa Hà Nội

## 📌 Mô tả dự án
Dự án xây dựng một **trợ lý ảo hỗ trợ tuyển sinh** cho Đại học Bách Khoa Hà Nội, ứng dụng các công nghệ trí tuệ nhân tạo hiện đại để trả lời câu hỏi tuyển sinh một cách nhanh chóng, chính xác và thân thiện.

## ⚙️ Công nghệ sử dụng
- **Python**
- **Streamlit** – Giao diện web thân thiện, dễ triển khai
- **ChromaDB** – Vector database phục vụ tìm kiếm ngữ nghĩa
- **LLM (Large Language Model)** – Mô hình ngôn ngữ lớn
- **RAG (Retrieval-Augmented Generation)** – Kỹ thuật tăng cường truy xuất

## 🧠 Tính năng chính
- Ứng dụng mô hình LLM kết hợp với kỹ thuật RAG để xây dựng chatbot có khả năng **truy xuất thông tin từ vector database** và trả lời câu hỏi tuyển sinh.
- **Pipeline xử lý dữ liệu**:
  - Thu thập dữ liệu từ website
  - Làm sạch nội dung
  - Chia nhỏ thông tin
  - Vector hóa dữ liệu với ChromaDB
  ![image](https://github.com/user-attachments/assets/f85e49b1-a8e6-4883-b0b5-674530bb96a5)

- **Fine-tuning mô hình LLM** để nâng cao độ chính xác và phù hợp với ngữ cảnh thực tế của người dùng (Trên bộ dữ liệu context, question, answer tự xây dựng)
- **Thiết kế UI bằng Streamlit**, thân thiện và dễ sử dụng.
![image](https://github.com/user-attachments/assets/e9a28cc9-0c8b-437e-9e6b-dc786ced63d4)


## 🚀 Mục tiêu
- Hỗ trợ thí sinh và phụ huynh tra cứu thông tin tuyển sinh chính xác, nhanh chóng.
- Cải thiện trải nghiệm tư vấn tuyển sinh bằng cách tự động hóa và cá nhân hóa thông tin.

## 📄 Tài liệu chi tiết
Slide thuyết trình:https://www.canva.com/design/DAGamG-Rj18/_Ww3E7vCArRcXwUCzg5uDg/edit?utm_content=DAGamG-Rj18&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
> 📌 *Dự án thuộc đồ án tốt nghiệp ngành Toán Tin, Đại học Bách Khoa Hà Nội.*
> Tác giả: Onion209

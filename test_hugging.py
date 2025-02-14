import torch
from unsloth import FastLanguageModel

# Cấu hình các tham số
max_seq_length = 2048  # Độ dài tối đa của chuỗi đầu vào
dtype = torch.bfloat16  # Kiểu dữ liệu để tối ưu bộ nhớ và tốc độ
load_in_4bit = True    # Lượng tử hóa 4-bit để giảm bộ nhớ

# Load model và tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "onionhust/Tuyen_sinh_model",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Bật chế độ inference nhanh hơn 2x
FastLanguageModel.for_inference(model)

# Chuyển model sang chế độ evaluation
model.eval()

# Nếu có GPU thì đưa model lên GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Hàm để sinh text
def generate_text(prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ví dụ sử dụng
# prompt = "Câu hỏi của bạn"
# response = generate_text(prompt)
# print(response)
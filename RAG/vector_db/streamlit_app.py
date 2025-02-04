import streamlit as st
from backend import ChatBackend
from main import classify_question_domain, load_vector_db, clear_gpu_memory
import os
from datetime import datetime
import logging  # Thêm import logging
import warnings

# Cấu hình logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Khởi tạo session state cho conversations
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None

# Khởi tạo backend trong session state
if 'backend' not in st.session_state:
    st.session_state.backend = ChatBackend()

# Thêm vào phần khởi tạo session state
if 'current_model_config' not in st.session_state:
    st.session_state.current_model_config = {
        'embedding_model': "OpenAI Embeddings",
        'llm_model': "gpt-4o-mini"
    }

# Tạo sidebar
with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSr2-hIfIOEB2-bok5hY83nxSQhmqOr0ANvTw&s",
            width=150
        )
    
    st.title("Model Configuration")
    
    # Chọn Embedding Model
    embedding_model = st.selectbox(
        "Choose Embedding Model",
        ["OpenAI Embeddings", "HuggingFace Embeddings"],
        index=0
    )
    
    # Chọn LLM Model
    llm_model = st.selectbox(
        "Choose LLM Model",
        ["gpt-4o-mini", "llama"],
        index=0,
        help="gpt-4o-mini uses OpenAI's API. llama uses a locally fine-tuned model."
    )
    
    # Nút để áp dụng cấu hình
    if st.button("Apply Configuration"):
        with st.spinner("Loading model... This may take a few moments."):
            try:
                st.session_state.backend.setup_qa_chain(
                    embedding_model_choice=embedding_model,
                    llm_model_choice=llm_model
                )
                # Lưu cấu hình hiện tại vào session state
                st.session_state.current_model_config = {
                    'embedding_model': embedding_model,
                    'llm_model': llm_model
                }
                st.success(f"""
                Configuration applied successfully:
                - Embedding Model: {embedding_model}
                - Language Model: {llm_model}
                """)
            except Exception as e:
                st.error(f"Error applying configuration: {str(e)}")
                logger.error(f"Configuration error: {str(e)}")

    # Phần quản lý hội thoại
    st.markdown("---")  # Đường kẻ phân cách
    st.subheader("Conversations")
    
    # Nút tạo cuộc trò chuyện mới
    if st.button("🆕 New Chat"):
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.conversations[new_id] = {
            "title": f"Chat {len(st.session_state.conversations) + 1}",
            "messages": [],
            "model_config": st.session_state.current_model_config.copy()  # Lưu cấu hình model cho chat này
        }
        st.session_state.current_conversation_id = new_id
        st.session_state.chat_history = []
        
        # Khởi tạo lại backend với model mới
        st.session_state.backend = ChatBackend()
        
        # Thiết lập model cho backend mới
        try:
            with st.spinner("Setting up model..."):
                st.session_state.backend.setup_qa_chain(
                    llm_model_choice=st.session_state.current_model_config['llm_model'],
                    embedding_model_choice=st.session_state.current_model_config['embedding_model']
                )
        except Exception as e:
            st.error(f"Error setting up model: {str(e)}")
            logger.error(f"Model setup error: {str(e)}")
        
        st.rerun()

    # CSS cho khu vực cuộn
    st.markdown("""
        <style>
            [data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
                max-height: 300px;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sử dụng expander để tạo khu vực có thể cuộn
    with st.expander("Chat History", expanded=True):
        for conv_id, conv_data in st.session_state.conversations.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📝 {conv_data['title']}", key=f"conv_{conv_id}"):
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.chat_history = conv_data['messages']
                    # Thiết lập lại backend với model config hiện tại
                    try:
                        st.session_state.backend.setup_qa_chain(
                            llm_model_choice=st.session_state.current_model_config['llm_model'],
                            embedding_model_choice=st.session_state.current_model_config['embedding_model']
                        )
                    except Exception as e:
                        st.error(f"Error setting up model: {str(e)}")
                        logger.error(f"Model setup error: {str(e)}")
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    del st.session_state.conversations[conv_id]
                    if st.session_state.current_conversation_id == conv_id:
                        st.session_state.current_conversation_id = None
                        st.session_state.chat_history = []
                    st.rerun()

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 16px; font-style: italic;'>One love, one future.</p>", 
        unsafe_allow_html=True
    )

# Main content
st.title("💬 HUST Admissions Consulting Assistant")

# Hiển thị tiêu đề cuộc trò chuyện hiện tại
if st.session_state.current_conversation_id:
    current_chat = st.session_state.conversations[st.session_state.current_conversation_id]
    st.subheader(f"Current Chat: {current_chat['title']}")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.markdown("**Nguồn tham khảo:**")
            for source in message["sources"]:
                st.markdown(f"- {source}")

# Xử lý input từ người dùng
user_input = st.chat_input("Hãy đặt câu hỏi về tuyển sinh...")

if user_input and st.session_state.current_conversation_id:
    # Hiển thị câu hỏi của user
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Xử lý câu trả lời
    with st.chat_message("assistant"):
        try:
            # Đảm bảo backend đang sử dụng đúng model
            current_chat = st.session_state.conversations[st.session_state.current_conversation_id]
            if ('model_config' not in current_chat or 
                current_chat['model_config'] != st.session_state.current_model_config):
                # Khởi tạo lại backend nếu model config không khớp
                st.session_state.backend = ChatBackend()
                st.session_state.backend.setup_qa_chain(
                    llm_model_choice=st.session_state.current_model_config['llm_model'],
                    embedding_model_choice=st.session_state.current_model_config['embedding_model']
                )
                current_chat['model_config'] = st.session_state.current_model_config.copy()
            
            # Sử dụng model config đã lưu trong session state
            result = st.session_state.backend.get_chat_response(
                user_input,
                model_choice=current_chat['model_config']['llm_model']
            )
            
            # Hiển thị câu trả lời
            st.markdown(result['answer'])
            
            # Hiển thị nguồn tham khảo
            if result['sources']:
                st.markdown("**Nguồn tham khảo:**")
                for source in result['sources']:
                    st.markdown(f"**{source['name']}**")
                    for vector in source['vectors']:
                        st.markdown(f"- {vector['vector']} (Score: {vector['score']})")
                        st.markdown(f"  *{vector['content']}*")
            
            # Lưu vào lịch sử
            message = {
                "role": "assistant",
                "content": result['answer'],
                "sources": [f"{os.path.basename(s['name'])} (Score: {s['score']})" for s in result['sources']]
            }
            st.session_state.chat_history.append(message)
            st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.chat_history

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logger.error(f"Error details: {str(e)}")

def on_shutdown():
    if 'backend' in st.session_state:
        st.session_state.backend.cleanup()

if __name__ == "__main__":
    try:
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='.*torch.classes.*')
        on_shutdown()
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")
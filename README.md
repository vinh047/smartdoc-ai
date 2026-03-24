# SmartDoc AI - RAG Document Processing Application 📄

Ứng dụng **SmartDoc AI** là một hệ thống Retrieval Augmented Generation (RAG) giúp bạn xử lý và tương tác với các tài liệu PDF một cách thông minh bằng AI.

---

## 🚀 Tính năng chính

- ✅ Upload và xử lý file PDF
- ✅ Tương tác với tài liệu thông qua AI chatbot
- ✅ Lưu lịch sử trò chuyện
- ✅ Vector Store để tìm kiếm nội dung liên quan
- ✅ Hỗ trợ múi thời gian và quản lý phiên làm việc

---

## 📋 Yêu cầu hệ thống

- **Python**: 3.9 hoặc cao hơn
- **Git**: Để clone project (nếu cần)
- **Ollama** (Tùy chọn): Nếu muốn sử dụng local LLM
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **Disk space**: Tối thiểu 2GB cho dependencies

---

## 💻 Cài đặt

### 1. Clone/Mở project
```bash
# Nếu chưa có project, clone từ repository
git clone <repository_url>
cd AI-docs

# Hoặc vào thư mục project đã tồn tại
```

### 2. Tạo virtual environment
```bash
# Trên Windows
python -m venv venv

# Kích hoạt virtual environment
venv\Scripts\activate

# Trên macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

> **Lưu ý**: Nếu bạn muốn sử dụng GPU để tăng tốc độ xử lý, cài đặt `faiss-gpu` thay vì `faiss-cpu`. Hãy xoá dòng `faiss-cpu` từ `requirements.txt` trước khi chạy lệnh trên.

### 4. Cấu hình (Tùy chọn)

Kiểm tra file cấu hình tại `src/config/settings.py` và điều chỉnh nếu cần:

```bash
# Thư mục dữ liệu
DATA_DIR = "data/"

# Đường dẫn Ollama (nếu sử dụng local LLM)
OLLAMA_URL = "http://localhost:11434"
```

---

## 🎯 Chạy ứng dụng

### Khởi động Streamlit App
```bash
streamlit run app.py
```
Ứng dụng sẽ mở tự động tại **http://localhost:8501**

### Hoặc khởi động từ Python trực tiếp
```bash
python app.py
```

---

## 📁 Cấu trúc Project

```
AI-docs/
├── app.py                 # Điểm vào chính (Streamlit app)
├── requirements.txt       # Python dependencies
├── README.md             # File hướng dẫn này
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/           # Cấu hình ứng dụng
│   │   ├── __init__.py
│   │   └── settings.py   # Các thiết lập chính
│   │
│   ├── core/             # Logic xử lý chính
│   │   ├── __init__.py
│   │   ├── database.py        # Quản lý database
│   │   ├── document_processor.py  # Xử lý PDF
│   │   └── rag_engine.py      # Engine RAG
│   │
│   └── ui/               # Giao diện người dùng
│       ├── __init__.py
│       └── components.py # Các component Streamlit
│
├── data/
│   ├── raw/              # Thư mục chứa PDF đầu vào
│   └── vector_store/     # Lưu trữ vector embeddings
│
└── __pycache__/          # Cache Python
```

---

## 🔧 Hướng dẫn sử dụng

### 1. **Upload tài liệu**
   - Nhấp vào phần "Upload file" ở giao diện chính
   - Chọn file PDF muốn xử lý
   - Chờ ứng dụng xử lý và tạo vector embeddings

### 2. **Chat với tài liệu**
   - Nhập câu hỏi trong ô chat ở cuối trang
   - AI sẽ trích xuất thông tin liên quan từ tài liệu
   - Xem kết quả và lịch sử trò chuyện

### 3. **Quản lý phiên trò chuyện**
   - Sidebar hiển thị tất cả các phiên làm việc
   - Nhấp để chuyển giữa các phiên
   - Tạo phiên mới nếu cần

---

## 🛠️ Cài đặt Ollama (Nếu sử dụng local LLM)

### 1. Tải Ollama
   - Truy cập [ollama.ai](https://ollama.ai)
   - Tải phiên bản phù hợp với OS của bạn

### 2. Cài đặt Ollama
   - Chạy installer và hoàn thành cài đặt

### 3. Kéo mô hình (Pull a model)
   ```bash
   ollama pull llama2
   # hoặc
   ollama pull neural-chat
   ```

### 4. Khởi chạy Ollama
   ```bash
   ollama serve
   ```
   Ollama sẽ lắng nghe tại `http://localhost:11434`

---

## ⚠️ Xử lý sự cố

### Lỗi: `ModuleNotFoundError: No module named 'streamlit'`
```bash
# Đảm bảo virtual environment được kích hoạt
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Cài đặt lại dependencies
pip install -r requirements.txt
```

### Lỗi: `ConnectionError: Cannot connect to Ollama`
- Kiểm tra Ollama có đang chạy: `ollama serve`
- Kiểm tra URL trong `src/config/settings.py` có đúng không
- Mặc định: `http://localhost:11434`

### Lỗi: `OutOfMemory` khi xử lý PDF lớn
- Giảm kích thước chunk size trong `src/core/document_processor.py`
- Hoặc sử dụng GPU nếu có: thay `faiss-cpu` bằng `faiss-gpu`

### Lỗi: Database không tìm thấy
- Database sẽ được tạo tự động lần đầu chạy
- Nếu lỗi vẫn tiếp tục, xoá file database và chạy lại:
```bash
# Database thường nằm ở: data/app.db
rm data/app.db
```

---

## 📚 Dependencies chính

| Package | Mục đích |
|---------|---------|
| **streamlit** | Frontend/UI framework |
| **langchain** | Orchestration LLM |
| **ollama** | Local LLM API |
| **faiss-cpu/gpu** | Vector similarity search |
| **transformers** | Embeddings models |
| **pdfplumber** | PDF parsing |
| **sqlite3** | Database (built-in) |

---

## 🤝 Đóng góp

Nếu bạn muốn cải thiện project:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push đến branch (`git push origin feature/YourFeature`)
5. Tạo Pull Request

---

## 📝 Ghi chú

- Lần chạy đầu tiên có thể chậm vì cần tải models
- Vector store lưu trữ tại `data/vector_store/` để tái sử dụng
- Chat history lưu trong database SQLite tại `data/app.db`

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:
- Kiểm tra lại các bước cài đặt
- Xem phần "Xử lý sự cố"
- Kiểm tra logs từ terminal

---

**Tạo ngày**: 2026-03-24  
**Phiên bản**: 1.0.0

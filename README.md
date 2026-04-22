# SmartDoc AI - Ứng dụng RAG Xử Lý Tài Liệu Thông Minh 📄

**SmartDoc AI** là một ứng dụng web thông minh giúp bạn upload các file tài liệu (PDF, Word), sau đó có thể tương tác với chúng thông qua một trợ lý AI. Ứng dụng sử dụng công nghệ **Retrieval Augmented Generation (RAG)** để tìm kiếm thông tin chính xác từ tài liệu của bạn và cung cấp câu trả lời dựa trên nội dung.

---

## 🚀 Các Tính Năng Chính

- 📤 **Upload tài liệu**: Hỗ trợ file PDF và Word (.docx)
- 🔍 **Tìm kiếm thông minh**: Kết hợp tìm kiếm từ khóa + tìm kiếm ngữ nghĩa (AI hiểu ngữ cảnh)
- 💬 **Chat với tài liệu**: Hỏi AI bất kỳ câu hỏi gì, AI sẽ trả lời dựa trên nội dung tài liệu
- 💾 **Lưu lịch sử**: Tất cả cuộc trò chuyện được lưu lại trong các "phiên làm việc"
- 📑 **Xử lý OCR**: Tự động nhận diện chữ từ hình ảnh trong PDF
- 🏷️ **Quản lý metadata**: Theo dõi nguồn gốc của thông tin (từ file nào, trang nào)
- ⚡ **Hiệu suất cao**: Sử dụng GPU để xử lý nhanh (nếu có)

---

## 📋 Yêu Cầu Hệ Thống

| Yêu cầu | Chi tiết |
|---------|---------|
| **Python** | Phiên bản 3.9 hoặc cao hơn |
| **RAM** | Tối thiểu 8GB (khuyến nghị 16GB) |
| **Disk space** | Tối thiểu 2-3GB cho dependencies + data |
| **GPU** (Tùy chọn) | RTX 3060 hoặc cao hơn để tăng tốc |
| **Ollama** (Nếu dùng LLM local) | Cần cài đặt riêng |

---

## 🔧 HƯỚNG DẪN CÀI ĐẶT (Chi Tiết Từng Bước)

### **Bước 1: Chuẩn bị thư mục project**

```bash
# Trên Windows PowerShell hoặc Windows Terminal
cd C:\Users\YourName\Documents
git clone <repository_url>
cd AI-docs
```

Hoặc nếu bạn đã tải code về, chỉ cần mở thư mục project.

### **Bước 2: Tạo môi trường ảo (Virtual Environment)**

**Tại sao cần:** Virtual environment giúp cách ly dependencies của project này, tránh xung đột với các project khác.

**Windows (PowerShell):**
```powershell
# Tạo môi trường ảo tên "venv"
python -m venv venv

# Kích hoạt môi trường
venv\Scripts\Activate.ps1

# Nếu gặp lỗi về execution policy, chạy:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

✅ **Xác nhận đã kích hoạt**: Bạn sẽ thấy `(venv)` ở đầu dòng terminal.

### **Bước 3: Cài đặt các thư viện (Dependencies)**

```bash
pip install -r requirements.txt
```

**Quá trình này sẽ cài đặt:**
- `streamlit`: Framework để tạo giao diện web
- `langchain`: Framework để xây dựng ứng dụng AI
- `ollama`: Kết nối với mô hình AI local
- `faiss-cpu`: Công cụ tìm kiếm vector (nhanh và chính xác)
- `sentence-transformers`: Chuyển đổi câu thành vector số
- `easyocr`: Nhận diện chữ từ hình ảnh
- Và các thư viện khác...

⏱️ **Thời gian**: Khoảng 5-15 phút tùy vào tốc độ internet.

### **Bước 4: Cấu hình ứng dụng (Tùy chọn)**

Mở file `src/config/settings.py` để điều chỉnh:

```python
# Cấu hình xử lý tài liệu
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,       # Số ký tự mỗi đoạn (tăng = xử lý nhanh, giảm = chính xác hơn)
    "chunk_overlap": 100      # Chồng lấp giữa các đoạn (đảm bảo không thiếu context)
}

# Cấu hình mô hình AI
LLM_CONFIG = {
    "model": "qwen2.5:7b",    # Mô hình AI sử dụng (thay đổi nếu muốn)
    "temperature": 0.7,        # 0-1, cao hơn = sáng tạo hơn, thấp hơn = chính xác hơn
    "top_p": 0.9,              # Lấy mẫu nucleus
    "repeat_penalty": 1.1      # Tránh lặp từ
}

# Cấu hình tìm kiếm
RETRIEVER_CONFIG = {
    "search_type": "similarity", # Hoặc "mmr" cho kết quả đa dạng
    "k": 3                       # Lấy 3 đoạn văn bản hàng đầu
}
```

---

## ✅ CHẠY ỨNG DỤNG

### **Cách 1: Chạy trên Streamlit (Khuyến Nghị)**

```bash
streamlit run app.py
```

**Điều gì xảy ra:**
1. Streamlit sẽ khởi động một server web local
2. Trình duyệt tự động mở ứng dụng tại `http://localhost:8501`
3. Bạn sẽ thấy giao diện web của SmartDoc AI

### **Cách 2: Chạy trực tiếp bằng Python**

```bash
python app.py
```

**Lưu ý**: Cách này chỉ dùng nếu không có Streamlit.

---

## 🎬 HƯỚNG DẪN SỬ DỤNG (Chi Tiết Luồng Hoạt Động)

### **Quy Trình Tổng Quát:**
```
Upload tài liệu → AI xử lý & lập chỉ mục → Tạo phiên chat → Đặt câu hỏi → AI trả lời
```

---

### **1️⃣ UPLOAD & XỬ LÝ TÀI LIỆU**

#### **Bước chi tiết:**

**A. Lên giao diện Upload**
- Tìm phần "📤 Upload File" ở cạnh trái (sidebar)
- Nhấp vào "Choose PDF or DOCX file"

**B. Chọn file**
- Chọn một file PDF hoặc Word
- Ấn "Upload" để tải lên

**C. AI xử lý tài liệu (Điều gì xảy ra phía sau)**

1. **Trích xuất text**: 
   - Đối với PDF: Đọc toàn bộ text từ mỗi trang
   - Nếu PDF chứa hình ảnh chữ: Dùng OCR (nhận diện ký tự) để trích xuất

2. **Chia nhỏ tài liệu**:
   - Tài liệu được chia thành các đoạn nhỏ (chunks) ~1000 ký tự
   - Các đoạn có chồng lấp nhau để giữ ngữ cảnh

   *Ví dụ*:
   ```
   Đoạn 1: "Công ty ABC thành lập năm 2020. Chúng tôi chuyên về..."
   Đoạn 2: "chuyên về phần mềm AI. Sản phẩm chính của chúng tôi..."
   Đoạn 3: "chính của chúng tôi bao gồm chatbot, phân tích dữ liệu..."
   ```

3. **Chuyển đổi thành vector (Embedding)**:
   - Mỗi đoạn text được chuyển thành một chuỗi số (vector)
   - Các số này có tính chất: **Các đoạn có ý nghĩa gần giống sẽ có vector gần giống**
   - Lưu vào FAISS (một cơ sở dữ liệu vector)

4. **Lưu metadata**:
   - Ghi lại: Tên file, số trang, thời gian upload
   - Dùng để truy vết thông tin sau

**✅ Hoàn thành**: Dòng thông báo "✓ Tài liệu đã được xử lý" sẽ xuất hiện

---

### **2️⃣ QUẢN LÝ PHIÊN CHAT**

#### **Bước chi tiết:**

**A. Tạo phiên mới**
- Nhấp "➕ Phiên làm việc mới" ở sidebar
- Nhập tên phiên (vd: "Phân tích hợp đồng", "Hỏi về sản phẩm")
- Chọn file tài liệu (có thể chọn nhiều file)

**B. Chuyển phiên**
- Danh sách phiên hiển thị ở sidebar
- Nhấp vào phiên bất kỳ để chuyển
- Lịch sử chat của phiên đó sẽ hiển thị

**C. Xoá phiên**
- Mở phiên → Nhấp "🗑️ Xoá phiên"
- Lịch sử chat sẽ bị xoá (không thể phục hồi)

---

### **3️⃣ ĐẶT CÂU HỎI & NHẬN CÂU TRẢ LỜI**

#### **Bước chi tiết:**

**A. Nhập câu hỏi**
- Tại dòng "Đặt câu hỏi về tài liệu...", gõ câu hỏi của bạn
- Ví dụ: "Công ty này làm gì?", "Giá sản phẩm là bao nhiêu?"

**B. AI tìm kiếm (Điều gì xảy ra phía sau)**

Ứng dụng sử dụng **Hybrid Retriever** (tìm kiếm kép):

1. **BM25 (Tìm kiếm từ khóa)**:
   - Tìm các đoạn có chứa từ khóa giống từ câu hỏi
   - Nhanh, chính xác cho từ khóa cụ thể
   - *Ví dụ*: Nếu hỏi "giá tiền", tìm những đoạn có chữ "giá" hoặc "tiền"

2. **Vector Search (Tìm kiếm ngữ nghĩa)**:
   - Chuyển câu hỏi thành vector
   - Tìm những đoạn có vector gần nhất
   - Hiểu được ý nghĩa, ngữ cảnh
   - *Ví dụ*: Nếu hỏi "bao nhiêu tiền", có thể hiểu là "giá bao nhiêu"

3. **Kết hợp**:
   - Lấy top 3 từ mỗi phương pháp
   - Kết hợp thành 6 đoạn tốt nhất
   - Loại bỏ trùng lặp

**C. AI sinh câu trả lời**
- Đưa 6 đoạn tốt nhất + câu hỏi vào mô hình ngôn ngữ (LLM)
- LLM đọc, hiểu, và viết câu trả lời
- Câu trả lời luôn dựa trên nội dung tài liệu

**D. Hiển thị kết quả**
- Câu trả lời hiển thị ở giữa màn hình
- Có thể nhấp để xem "Nguồn tham khảo" (đoạn nào được dùng)

---

### **4️⃣ LỮU VÀ XEM LỊCH SỬ**

- Mỗi câu hỏi & câu trả lời được lưu tự động vào database
- Lịch sử của phiên hiện tại hiển thị trên cùng lúc
- Đóng ứng dụng & mở lại, dữ liệu vẫn còn

---

## 📁 CẤU TRÚC THỐNG THÀNH (Project Structure)

```
AI-docs/
│
├── 📄 app.py                          # ⭐ File chính - Điểm bắt đầu của ứng dụng
│                                      # Nạp tất cả modules, khởi tạo Streamlit UI
│
├── 📋 requirements.txt                # Danh sách các thư viện cần cài
├── 📖 README.md                       # File hướng dẫn này
│
├── 📁 src/                            # Thư mục chứa mã chính (Source code)
│   │
│   ├── 📁 config/                     # ⚙️ CẤU HÌNH
│   │   ├── settings.py                # Các tham số: model, chunk size, k, temperature
│   │   └── __init__.py
│   │
│   ├── 📁 core/                       # 🔧 LÕI CHÍNH - LOGIC XỬ LÝ
│   │   ├── database.py                # Lưu/tải chat history từ SQLite
│   │   ├── document_processor.py      # Xử lý PDF & DOCX (trích text, cắt chunks)
│   │   ├── vector_manager.py          # Tạo vector store FAISS, lưu embeddings
│   │   ├── rag_engine.py              # Hybrid retriever (BM25 + vector search)
│   │   ├── metadata_handler.py        # Quản lý metadata (source, page, ...)
│   │   ├── ocr_utils.py               # OCR - nhận diện chữ từ hình ảnh PDF
│   │   │
│   │   └── 📁 engines/                # Các engine nâng cao
│   │       └── advanced_rag.py        # Pipeline RAG chi tiết
│   │
│   ├── 📁 ui/                         # 🎨 GIAO DIỆN NGƯỜI DÙNG
│   │   ├── components.py              # Components Streamlit (header, sidebar, ...)
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── 📁 data/                           # 📊 DỮ LIỆU
│   ├── raw/                           # PDF/DOCX upload vào đây
│   ├── temp/                          # File tạm khi đang xử lý
│   ├── images/                        # Hình ảnh trích từ PDF
│   └── vector_store/                  # FAISS vector store (lưu embeddings)
│
├── 📁 tests/                          # ✅ KIỂM THỬ
│   ├── conftest.py
│   ├── test_document_processor.py     # Test xử lý tài liệu
│   ├── test_rag_engine.py             # Test RAG engine
│   ├── test_vector_manager.py         # Test vector search
│   └── ...
│
├── 📁 venv/                           # 🐍 VIRTUAL ENVIRONMENT
│   └── (Nằm ở đây sau khi chạy: python -m venv venv)
│
└── .env (Tùy chọn)                   # Biến môi trường (API keys, URLs, ...)
```

---

## 🤖 CÁCH HOẠT ĐỘNG CỦA AI (Luồng Chi Tiết)

### **Sơ đồ hoạt động:**

```
┌─────────────────────────────────────────────────────────────┐
│                    NGƯỜI DÙNG HỎI CÂU HỎI                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  1️⃣ CHUYỂN VÀO VECTOR KHÔNG GIAN           │
│  "Công ty làm gì?" → [0.23, -0.45, 0.67, 0.12, ..., 0.89] │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         ▼                                ▼
┌──────────────────────┐        ┌──────────────────────┐
│   2️⃣ BM25 TỚM       │        │  3️⃣ VECTOR SEARCH  │
│ Tìm từ khóa "gì"    │        │  Tìm vector gần     │
│ → Top 3 đoạn        │        │  → Top 3 đoạn       │
│                      │        │                      │
│ Đoạn A              │        │ Đoạn D              │
│ Đoạn B              │        │ Đoạn E              │
│ Đoạn C              │        │ Đoạn F              │
└──────────────────────┘        └──────────────────────┘
         │                                │
         └───────────────┬────────────────┘
                         │
                         ▼
         ┌──────────────────────────────┐
         │  4️⃣ KẾT HỢP 6 ĐOẠN TỐT NHẤT │
         │  (Loại trùng, xắp xếp)       │
         │                              │
         │  Context: [Đoạn A, D, B,...]  │
         │  Question: "Công ty làm gì?"  │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │ 5️⃣ ĐỀ VÀO MÔ HÌNH NGÔN NGỮ  │
         │    (Ollama - qwen2.5:7b)     │
         │                              │
         │ Prompt: "Dựa vào thông tin   │
         │ dưới đây, trả lời câu hỏi:   │
         │ [Context]                    │
         │ [Question]"                  │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │  6️⃣ AI SINH TÁ CÂU TRẢ LỜI  │
         │                              │
         │  "Công ty ABC chuyên về      │
         │   phần mềm AI. Sản phẩm      │
         │   chính bao gồm... "         │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │  7️⃣ HIỂN THỊ + LƯU DỮ LIỆU  │
         │  - Câu trả lời trên giao diện│
         │  - Lưu vào database SQLite   │
         │  - Hiển thị nguồn tham khảo  │
         └──────────────────────────────┘
```

---

## 🛠️ CÀI ĐẶT OLLAMA (Để chạy mô hình AI Local)

**Ollama** là một công cụ cho phép chạy mô hình AI lớn (LLM) trên máy tính của bạn, mà không cần kết nối internet hay tài khoản API.

### **Bước 1: Tải Ollama**
- Truy cập: https://ollama.ai
- Tải phiên bản phù hợp OS của bạn (Windows, Mac, Linux)

### **Bước 2: Cài đặt**
- Chạy file installer
- Làm theo hướng dẫn trên màn hình

### **Bước 3: Tải mô hình (Pull model)**

**Trong PowerShell hoặc Terminal:**
```bash
# Tải mô hình Qwen (mặc định, nên hơn)
ollama pull qwen2.5:7b

# Hoặc Llama 2 (chậm hơn, nhưng tốt)
ollama pull llama2

# Hoặc Mistral (nhanh, sáng tạo)
ollama pull mistral
```

**Lần đầu sẽ tải ~3-4GB**, phụ thuộc vào mô hình.

### **Bước 4: Khởi chạy Ollama**

```bash
ollama serve
```

- Ollama sẽ khởi động ở `http://localhost:11434`
- Giữ terminal này mở, không đóng

### **Bước 5: Kiểm tra kết nối**

Trong PowerShell khác:
```bash
curl http://localhost:11434/api/tags
```

Nếu thấy danh sách model, Ollama đang chạy ✅

---

## ⚠️ XỬ LÝ SỰ CỐ (Troubleshooting)

### **Lỗi 1: `ModuleNotFoundError: No module named 'streamlit'`**

**Nguyên nhân**: Chưa cài thư viện hoặc chưa kích hoạt venv

**Cách sửa**:
```bash
# Kiểm tra venv có kích hoạt (sẽ thấy (venv) ở đầu dòng)
venv\Scripts\activate  # Windows

# Cài đặt lại
pip install streamlit
```

---

### **Lỗi 2: `ConnectionError: Cannot connect to Ollama`**

**Nguyên nhân**: Ollama chưa chạy hoặc URL sai

**Cách sửa**:
```bash
# Terminal 1 - Chạy Ollama
ollama serve

# Terminal 2 - Chạy app
streamlit run app.py

# Kiểm tra URL trong src/config/settings.py:
# OLLAMA_URL = "http://localhost:11434"  # Đúng chưa?
```

---

### **Lỗi 3: `OutOfMemory` khi xử lý PDF lớn**

**Nguyên nhân**: RAM không đủ hoặc PDF quá lớn

**Cách sửa**:
```python
# Mở src/config/settings.py, giảm chunk_size:
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,        # ← Giảm từ 1000 xuống 500
    "chunk_overlap": 50       # ← Giảm từ 100 xuống 50
}
```

**Hoặc**:
- Dùng GPU (nếu có): Thay `faiss-cpu` bằng `faiss-gpu` trong requirements.txt
- Cập nhật RAM hoặc đóng các chương trình khác

---

### **Lỗi 4: Không thể upload file hoặc file không được xử lý**

**Kiểm tra**:
```bash
# 1. Thư mục data/raw có tồn tại không?
ls data/

# 2. File có quyền ghi không?
# 3. Format file có đúng (PDF/DOCX)?
# 4. Xem terminal của Streamlit có lỗi nào không
```

---

### **Lỗi 5: Chat history bị mất sau khi tắt ứng dụng**

**Nguyên nhân**: Database chưa được khởi tạo

**Cách sửa**:
```bash
# Xoá database cũ (nếu bị hỏng)
rm data/app.db

# Chạy lại app - database sẽ được tạo mới
streamlit run app.py
```

---

## 📚 Các Thư Viện Chính

| Thư viện | Tác dụng |
|---------|---------|
| **Streamlit** | Tạo giao diện web đẹp (không cần JavaScript) |
| **LangChain** | Framework xây dựng ứng dụng AI, quản lý prompt |
| **Ollama** | Chạy mô hình AI local, không cần API |
| **FAISS** | Tìm kiếm vector cực nhanh (CPU hoặc GPU) |
| **Sentence-Transformers** | Chuyển text → vector (embedding) |
| **EasyOCR** | Nhận diện chữ từ ảnh trong PDF |
| **PyMuPDF (fitz)** | Đọc file PDF, trích text & ảnh |
| **python-docx** | Đọc file Word (.docx) |
| **SQLite3** | Database lưu chat history (có sẵn Python) |

---

## 💡 TIPS & TRICKS

✅ **Để tăng độ chính xác**:
- Upload tài liệu nguyên bản (chứ không phải scan)
- Tài liệu có tóm tắt/ đoạn mở đầu sẽ được xử lý tốt hơn

✅ **Để tăng tốc độ**:
- Giảm `chunk_size` trong cấu hình
- Sử dụng GPU nếu có (`faiss-gpu`)
- Tăng `k` (số lượng kết quả lấy ra) - không, giảm lại!

✅ **Để tối ưu hóa RAM**:
- Xoá vector_store cũ: `rm -r data/vector_store/`
- Upload từng file một, chứ không upload hàng loạt

---

## 📞 Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề:
1. ✅ Kiểm tra lại các bước cài đặt
2. ✅ Xem phần "Xử lý sự cố" ở trên
3. ✅ Kiểm tra logs ở terminal
4. ✅ Tìm kiếm error message trên Google / Stack Overflow

---

## 📝 Ghi Chú Quan Trọng

- ⏱️ **Lần chạy đầu tiên**: Chậm vì cần tải models, embeddings (chờ 2-5 phút)
- 💾 **Vector store**: Lưu tại `data/vector_store/`, tái sử dụng được (nhanh lần sau)
- 💬 **Chat history**: Lưu trong SQLite (`data/app.db`), không bị mất
- 🔄 **Reuse**: Có thể upload tài liệu giống nhau không, sẽ dùng lại vector cũ

---

**Phiên bản**: 1.0.0  
**Cập nhật lần cuối**: 2026-04-22

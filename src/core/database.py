import sqlite3
import uuid
import os

# Đảm bảo thư mục data tồn tại để lưu file DB
os.makedirs("data", exist_ok=True)
DB_PATH = "data/chat_history.db"

def init_db():
    """Khởi tạo cơ sở dữ liệu và các bảng (Chạy 1 lần khi bật app)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Bảng Sessions: Lưu thông tin các phiên trò chuyện
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. Bảng Messages: Lưu chi tiết từng tin nhắn thuộc về phiên nào
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT, -- 'user' hoặc 'assistant'
            content TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    conn.commit()
    conn.close()

def create_new_session(title="Cuộc trò chuyện mới"):
    """Tạo một phiên chat mới và trả về ID"""
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO sessions (id, title) VALUES (?, ?)', (session_id, title))
    conn.commit()
    conn.close()
    return session_id

def get_all_sessions():
    """Lấy danh sách tất cả phiên chat để hiển thị ra Sidebar"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, title, created_at FROM sessions ORDER BY created_at DESC')
    sessions = c.fetchall()
    conn.close()
    # Chuyển đổi tuple thành list of dicts cho dễ dùng
    return [{"id": row[0], "title": row[1], "created_at": row[2]} for row in sessions]

def add_message(session_id, role, content):
    """Lưu một tin nhắn mới vào DB"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', 
              (session_id, role, content))
    conn.commit()
    conn.close()

def get_messages_by_session(session_id):
    """Lấy toàn bộ lịch sử tin nhắn của một phiên cụ thể"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC', (session_id,))
    messages = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in messages]
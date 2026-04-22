import sqlite3
import uuid
import os
import shutil

# --- CẤU HÌNH HỆ THỐNG ---
# Đảm bảo thư mục data luôn tồn tại để lưu trữ cơ sở dữ liệu
os.makedirs("data", exist_ok=True)
DB_PATH = "data/chat_history.db"


def init_db():
    """
    Khởi tạo cấu trúc cơ sở dữ liệu SQLite.
    Tạo các bảng 'sessions' và 'messages' nếu chúng chưa tồn tại.
    Hàm này nên được gọi một lần duy nhất khi ứng dụng bắt đầu khởi chạy.
    """
    with sqlite3.connect(DB_PATH) as conn:
        # Bảng sessions: Lưu trữ thông tin định danh của các cuộc trò chuyện
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,           -- Mã định danh duy nhất (UUID)
                title TEXT,                    -- Tiêu đề của phiên chat
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Thời gian tạo
            )
        """
        )

        # Bảng messages: Lưu trữ chi tiết lịch sử tin nhắn của từng phiên
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, -- ID tự tăng
                session_id TEXT,                      -- Liên kết với bảng sessions
                role TEXT,                             -- Vai trò: 'user' hoặc 'assistant'
                content TEXT,                          -- Nội dung tin nhắn (có thể là JSON cho Split-screen)
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """
        )


def create_new_session(title: str = "Cuộc trò chuyện mới") -> str:
    """
    Tạo một phiên làm việc (session) mới trong cơ sở dữ liệu.

    Args:
        title (str): Tiêu đề hiển thị của phiên chat.

    Returns:
        str: ID của phiên vừa tạo (chuỗi UUID).
    """
    session_id = str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions (id, title) VALUES (?, ?)", (session_id, title)
        )
    return session_id


def get_all_sessions() -> list:
    """
    Truy vấn danh sách tất cả các phiên trò chuyện hiện có.
    Dùng để hiển thị danh sách các cuộc hội thoại cũ trên Sidebar.

    Returns:
        list: Danh sách các dictionary chứa thông tin id, title và ngày tạo.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
        )
        sessions = cursor.fetchall()

    return [{"id": row[0], "title": row[1], "created_at": row[2]} for row in sessions]


def add_message(session_id: str, role: str, content: str):
    """
    Lưu trữ một tin nhắn mới vào lịch sử của một phiên cụ thể.

    Args:
        session_id (str): ID của phiên chat cần lưu tin nhắn.
        role (str): Vai trò của người gửi ('user' hoặc 'assistant').
        content (str): Nội dung văn bản hoặc chuỗi JSON kết quả RAG.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_messages_by_session(session_id: str) -> list:
    """
    Lấy lại toàn bộ lịch sử tin nhắn của một phiên chat dựa trên ID.

    Args:
        session_id (str): ID của phiên chat cần truy xuất.

    Returns:
        list: Danh sách các dictionary chứa role và nội dung tin nhắn.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        messages = cursor.fetchall()

    return [{"role": row[0], "content": row[1]} for row in messages]


def clear_history(session_id: str):
    """
    Xóa sạch tất cả tin nhắn trong một phiên chat nhưng vẫn giữ lại tên phiên.

    Args:
        session_id (str): ID của phiên chat cần dọn dẹp.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def delete_session_permanently(session_id: str):
    """
    Xóa vĩnh viễn một phiên trò chuyện bao gồm cả tiêu đề và toàn bộ tin nhắn liên quan.
    Tương đương với chức năng 'Delete Chat' trong Gemini/ChatGPT.

    Args:
        session_id (str): ID của phiên chat cần xóa bỏ hoàn toàn.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Xóa tin nhắn trước để đảm bảo tính toàn vẹn (Foreign Key)
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            # Xóa thông tin phiên chính
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    except Exception as e:
        print(f"Lỗi hệ thống khi xóa phiên {session_id}: {e}")


def clear_vector_store() -> bool:
    """
    Xóa bỏ toàn bộ kho dữ liệu Vector (FAISS index) trên đĩa cứng.
    Dùng khi người dùng muốn reset lại hoàn toàn hệ thống tài liệu.

    Returns:
        bool: True nếu xóa thành công hoặc thư mục không tồn tại, False nếu có lỗi.
    """
    faiss_path = "data/faiss_index"
    if os.path.exists(faiss_path):
        try:
            shutil.rmtree(faiss_path)
            return True
        except Exception as e:
            print(f"Lỗi khi xóa Vector Store: {e}")
            return False
    return True

def update_session_title(session_id: str, new_title: str):
    """Cập nhật tiêu đề phiên chat dựa trên câu hỏi đầu tiên của người dùng"""
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
    except Exception as e:
        print(f"Lỗi update title: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
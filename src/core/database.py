import os
import json
import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    Text,
    event,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.engine import Engine
from src.config import DB_PATH

# Tạo thư mục data nếu chưa có
os.makedirs("data", exist_ok=True)


# Kích hoạt WAL mode cho SQLite để tối ưu tốc độ đọc/ghi đồng thời
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Khởi tạo SQLAlchemy Engine
engine = create_engine(DB_PATH, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 1. ĐỊNH NGHĨA CÁC MODEL (TABLES)
# ==========================================


class Document(Base):
    """Quản lý file gốc"""

    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=True)
    language = Column(String, default="vi")
    meta_data = Column(Text, default="{}")
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Lưu văn bản thuần"""

    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    """Quản lý phiên hội thoại"""

    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )
    retrieval_logs = relationship(
        "RetrievalLog", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """Quản lý tin nhắn"""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    meta_data = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class RetrievalLog(Base):
    """Nhật ký truy xuất"""

    __tablename__ = "retrieval_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    query = Column(Text, nullable=False)
    retrieved_chunks = Column(Text, default="[]")
    scores = Column(Text, default="[]")
    is_relevant = Column(Boolean, default=True)
    used_fallback = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="retrieval_logs")


# ==========================================
# 2. CÁC HÀM CRUD CƠ BẢN
# ==========================================


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_new_session(title="Phiên làm việc mới"):
    db = SessionLocal()
    try:
        new_session = ChatSession(session_name=title)
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return new_session.id
    finally:
        db.close()


def get_all_sessions():
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
        return [
            {"id": s.id, "title": s.session_name, "created_at": str(s.created_at)}
            for s in sessions
        ]
    finally:
        db.close()


def add_message(session_id, role, content, meta_data=None):
    db = SessionLocal()
    try:
        msg_meta = json.dumps(meta_data) if meta_data else "{}"
        new_msg = ChatMessage(
            session_id=session_id, role=role, message=content, meta_data=msg_meta
        )
        db.add(new_msg)

        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.updated_at = datetime.utcnow()

        db.commit()
    finally:
        db.close()


def get_messages_by_session(session_id):
    db = SessionLocal()
    try:
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        return [
            {"role": m.role, "content": m.message, "meta_data": json.loads(m.meta_data)}
            for m in messages
        ]
    finally:
        db.close()


def delete_session(session_id):
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
    finally:
        db.close()


def log_retrieval(
    session_id,
    query,
    retrieved_chunks_ids,
    scores,
    is_relevant=True,
    used_fallback=False,
):
    db = SessionLocal()
    try:
        log = RetrievalLog(
            session_id=session_id,
            query=query,
            retrieved_chunks=json.dumps(retrieved_chunks_ids),
            scores=json.dumps(scores),
            is_relevant=is_relevant,
            used_fallback=used_fallback,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()


# ==========================================
# 3. CÁC HÀM NÂNG CAO & QUẢN LÝ TÀI LIỆU
# ==========================================


def get_uploaded_files():
    """Lấy danh sách TÊN các file ĐÃ TẢI LÊN"""
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        return [doc.file_name for doc in docs]
    finally:
        db.close()


def get_uploaded_files_info():
    """Lấy thông tin CHI TIẾT các file để hiển thị UI"""
    db = SessionLocal()
    try:
        docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
        return [
            {
                "id": doc.id,
                "file_name": doc.file_name,
                "uploaded_at": doc.uploaded_at.strftime("%Y-%m-%d %H:%M"),
                "file_size": doc.file_size,
            }
            for doc in docs
        ]
    finally:
        db.close()


def delete_document(file_name):
    """Xóa một tài liệu cụ thể theo tên file"""
    db = SessionLocal()
    try:
        docs = db.query(Document).filter(Document.file_name == file_name).all()
        for doc in docs:
            db.delete(doc)
        db.commit()
    finally:
        db.close()


def clear_all_documents():
    """Xóa sạch Database tài liệu"""
    db = SessionLocal()
    try:
        db.query(Document).delete()
        db.commit()
    finally:
        db.close()


def rename_session(session_id, new_title):
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.session_name = new_title
            db.commit()
    finally:
        db.close()


def clear_all_sessions():
    db = SessionLocal()
    try:
        db.query(ChatSession).delete()
        db.commit()
    finally:
        db.close()

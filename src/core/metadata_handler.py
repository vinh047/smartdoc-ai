import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
from langchain_core.documents import Document


class MetadataManager:
    """Quản lý metadata cho các tài liệu được upload."""

    def create_metadata(self, filename: str, doc_type: str = "document") -> Dict[str, Any]:
        """
        Tạo một dictionary metadata chuẩn cho tài liệu.

        Args:
            filename: Tên file gốc của tài liệu.
            doc_type: Loại tài liệu (mặc định: 'document').

        Returns:
            Dict chứa source, upload_date, doc_type, và doc_id.
        """
        return {
            "source": filename,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "doc_type": doc_type,
            "doc_id": str(uuid.uuid4()),
        }

    def attach_metadata_to_docs(
        self, documents: List[Document], metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Gắn metadata vào từng Document object trong danh sách.
        Giữ lại metadata gốc từ loader (ví dụ: page number) và ghi đè/thêm
        các trường từ metadata mới.

        Args:
            documents: Danh sách LangChain Document objects.
            metadata: Dictionary metadata cần gắn vào.

        Returns:
            Danh sách Document đã được cập nhật metadata.
        """
        for doc in documents:
            doc.metadata.update(metadata)
        return documents

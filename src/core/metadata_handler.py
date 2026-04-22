import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
from langchain_core.documents import Document


class MetadataManager:
    """
    Quản lý và đồng bộ metadata cho các tài liệu được tải lên hệ thống.
    Hỗ trợ việc phân loại, lọc và truy xuất nguồn gốc tài liệu trong Vector Database.
    """

    def create_metadata(
        self, filename: str, doc_type: str = "document"
    ) -> Dict[str, Any]:
        """
        Tạo một bộ metadata chuẩn cho tài liệu mới.

        Args:
            filename (str): Tên file gốc của tài liệu (VD: 'baocao.pdf').
            doc_type (str, optional): Phân loại tài liệu. Mặc định là 'document'.

        Returns:
            Dict[str, Any]: Chứa thông tin về nguồn (source), ngày tải lên (upload_date),
                            loại (doc_type) và mã định danh duy nhất (doc_id).
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
        Gắn hoặc ghi đè metadata vào danh sách các đoạn văn bản (chunks).
        Lưu ý: Các metadata có sẵn từ bộ đọc (như page number) vẫn sẽ được giữ nguyên.

        Args:
            documents (List[Document]): Danh sách các object Document của LangChain.
            metadata (Dict[str, Any]): Bộ metadata chuẩn cần gắn vào.

        Returns:
            List[Document]: Danh sách tài liệu đã được cập nhật metadata.
        """
        for doc in documents:
            doc.metadata.update(metadata)
        return documents

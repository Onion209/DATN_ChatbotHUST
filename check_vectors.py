from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List, Optional

class ChromaQuery:
    def __init__(self, collection):
        self.collection = collection

    def query_documents(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None
    ) -> List[Documents]:
        """
        Thực hiện truy vấn documents dựa trên câu query đầu vào
        
        Args:
            query_text (str): Câu truy vấn
            n_results (int): Số lượng kết quả trả về
            where (dict, optional): Điều kiện lọc metadata
            where_document (dict, optional): Điều kiện lọc document
            
        Returns:
            List[Documents]: Danh sách documents phù hợp nhất
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            return results
            
        except Exception as e:
            print(f"Lỗi khi truy vấn: {str(e)}")
            return []

    def similarity_search(
        self,
        query_text: str,
        n_results: int = 5
    ) -> List[Documents]:
        """
        Tìm kiếm các documents tương tự dựa trên độ tương đồng vector
        
        Args:
            query_text (str): Câu truy vấn
            n_results (int): Số lượng kết quả trả về
            
        Returns:
            List[Documents]: Danh sách documents có độ tương đồng cao nhất
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            return results["documents"][0]
            
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {str(e)}")
            return []
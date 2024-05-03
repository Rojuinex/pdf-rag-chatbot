import uuid
import hashlib
from io import StringIO
from typing import Optional

from duckdb import DuckDBPyConnection

from pdfminer.converter import TextConverter as PDFMinerTextConverter
from pdfminer.layout import LAParams as PDFMinerLAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import (
    PDFResourceManager,
    PDFPageInterpreter
)
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


from pdf_rag_chatbot.db.models import (
    UploadedFile,
    Document,
)
from pdf_rag_chatbot.data_pipeline.steps.pipeline_step import PipelineStep
from pdf_rag_chatbot.data_pipeline.messages import (
    FileUploaded,
    DocumentCreated,
)


class Ingest(PipelineStep):
    def __init__(self, db: DuckDBPyConnection):
        super().__init__(
            "ingest",
            db=db,
            request_type=FileUploaded,
        )

    def __call__(self, req: FileUploaded) -> Optional[Document]:
        assert req.file_path.split(".")[-1].lower() in ["pdf", "txt", "text"], "Invalid file extension."

        if req.file_path.endswith(".pdf"):
            text = self._extract_text_from_pdf(req.file_path)
        else:
            with open(req.file_path, "r") as f:
                text = f.read()
        
        document_hash = hashlib.sha256(text.encode()).hexdigest()

        uploaded_file = UploadedFile(
            file_uuid=str(uuid.uuid4()),
            file_path=req.file_path,
            document_hash=document_hash,
            session_id=req.session_id,
        )

        self.db.execute(
            """--sql
                INSERT INTO uploaded_file (
                    file_uuid,
                    file_path,
                    document_hash,
                    session_id,
                    uploaded_at
                )
                VALUES (?, ?, ?, ?, ?)
            """,
            (
                uploaded_file.file_uuid,
                uploaded_file.file_path,
                uploaded_file.document_hash,
                uploaded_file.session_id,
                uploaded_file.uploaded_at,
            ),
        )

        # Check if the document has already been processed
        res = self.db.execute(
            "SELECT COUNT(*) > 0 AS is_processed FROM document WHERE document_hash = ?",
            (document_hash,),
        ).fetchone()

        if res[0] == True:
            return None

        document = Document(
            document_hash=document_hash,
            text=text,
        )

        self.db.execute(
            """--sql
                INSERT INTO document (
                    document_hash,
                    text,
                    processed_at
                )
                VALUES (?, ?, ?)
            """,
            (
                document.document_hash,
                document.text,
                document.processed_at,
            ),
        )

        return DocumentCreated(
            document=document
        )

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file.

        Args:
            file_path (str): The name of the PDF file.

        Returns:
            str: The extracted text.
        """

        output_string = StringIO()
        with open(file_path, 'rb') as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = PDFMinerTextConverter(
                rsrcmgr,
                output_string,
                laparams=PDFMinerLAParams(),
            )
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)

        return output_string.getvalue()

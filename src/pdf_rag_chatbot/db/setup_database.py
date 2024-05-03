from duckdb import DuckDBPyConnection

def setup_database(db: DuckDBPyConnection):
    """Initialize the database.

    The database schema is as follows:

                           ┌───────────────┐                 
                           │ Uploaded File │                 
                           └───────────────┘                 
                                  ╲│╱                        
                                   │                         
                                   ┼                         
                            ┌────────────┐                   
                     ┌─────┼│  Document  │┼─────┐            
                     │      └────────────┘      │            
                    ╱│╲                        ╱│╲           
           ┌───────────────────┐       ┌────────────────┐    
           │ Document Sentence │       │Document Entity │    
           └───────────────────┘       └────────────────┘    
                    ╲│╱                        ╲│╱           
                     │                          │            
                     ┼                          ┼            
               ┌───────────┐               ┌────────┐        
               │ Sentence  │               │ Entity │        
               └───────────┘               └────────┘        
                     ┼                          ┼            
                     │                          │            
                     │   ╱┌────────────────┐╲   │            
                     └────│ Text Embedding │────┘            
                         ╲└────────────────┘╱                
                                                             

    Uploaded File: Represents a file uploaded by the user.
    - `file_uuid`: A unique identifier for the file.
    - `file_path`: The path to the file.
    - `document_hash`: The sha256 hash of the document.
    - `session_id`: The session ID the file was uploaded in,
                    or NULL if the file was processed outside
                    of a user session.
    - `uploaded_at`: The timestamp of when the file was uploaded.

    Document: Represents a unique document.

    - `document_hash`: The sha256 hash of the document.
    - `processed_at`: The timestamp of when the document was processed.

    Sentence: Represents a unique sentence.

    - `cased_sentence_hash`: The md5 hash of the cased sentence.
    - `uncased_sentence_hash`: The md5 hash of the uncased sentence.
    - `text`: The text of the sentence.
    - `processed_at`: The timestamp of when the sentence was processed.

    Entity: Represents a unique entity.

    - `cased_entity_hash`: The md5 hash of the cased entity.
    - `uncased_entity_hash`: The md5 hash of the uncased entity.
    - `text`: The text of the entity.
    - `label`: The label of the entity.
    - `processed_at`: The timestamp of when the entity was processed.

    Document Sentence: Represents a sentence in a document.

    - `document_hash`: The sha256 hash of the document.
    - `cased_sentence_hash`: The md5 hash of the cased sentence.
    - `uncased_sentence_hash`: The md5 hash of the uncased sentence.
    - `index`: The index of the sentence in the document.
    - `start_char`: The starting character of the sentence in the document.
    - `end_char`: The ending character of the sentence in the document.
    - `processed_at`: The timestamp of when the sentence was processed.

    Document Entity: Represents an entity in a document.
    
    - `document_hash`: The sha256 hash of the document.
    - `cased_entity_hash`: The md5 hash of the cased entity.
    - `uncased_entity_hash`: The md5 hash of the uncased entity.
    - `start_char`: The starting character of the entity in the document.
    - `end_char`: The ending character of the entity in the document.
    - `timestamp`: The timestamp of when the entity was processed.

    Args:
        db (DuckDBPyConnection): The DuckDB connection.

    Raises:
        Exception: If the database connection fails.
    """

    db.execute(
        """--sql
            CREATE TABLE IF NOT EXISTS uploaded_file (
                file_uuid STRING PRIMARY KEY,
                file_path STRING NOT NULL,
                document_hash STRING NOT NULL,
                session_id STRING,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document (
                document_hash STRING PRIMARY KEY,
                text STRING NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sentence (
                cased_sentence_hash STRING PRIMARY KEY,
                uncased_sentence_hash STRING NOT NULL,
                text STRING NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS entity (
                cased_entity_hash STRING PRIMARY KEY,
                uncased_entity_hash STRING NOT NULL,
                text STRING NOT NULL,
                label STRING NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_sentence (
                document_hash STRING NOT NULL,
                cased_sentence_hash STRING NOT NULL,
                uncased_sentence_hash STRING NOT NULL,
                index INTEGER NOT NULL,
                text STRING NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_hash) REFERENCES document(document_hash)
            );

            CREATE TABLE IF NOT EXISTS document_entity (
                document_hash STRING NOT NULL,
                cased_entity_hash STRING NOT NULL,
                uncased_entity_hash STRING NOT NULL,
                text STRING NOT NULL,
                sentence_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                label STRING NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_hash) REFERENCES document(document_hash)
            );

            CREATE TABLE IF NOT EXISTS text_embedding (
                cased_text_hash STRING NOT NULL,
                uncased_text_hash STRING NOT NULL,
                model_name STRING NOT NULL,
                text STRING NOT NULL,
                embedding FLOAT[] NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (cased_text_hash, model_name),
            );
        """
    )


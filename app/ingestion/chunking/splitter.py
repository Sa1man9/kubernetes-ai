import logfire

def chunk_text(text:str, chunk_size:int=1500)->list[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size.
    """
    with logfire.span("Chunking Text",text_length=len(text)):
        if not text.strip():
            logfire.warning("⚠️ Received empty text for chunking.")
            return []
    
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) <= chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [chunk for chunk in chunks if chunk.strip()]

        if not valid_chunks:
            logfire.warning("⚠️ No valid chunks were created from the text.")
        
        logfire.info(f"✅ Chunking complete. Created {len(valid_chunks)} chunks.")

        return valid_chunks

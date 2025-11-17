"""
Enhanced Document Ingestion with Semantic/Hierarchical Chunking
Preserves legal document structure (Act → Section → Clause)
"""

import re
import json
import pickle
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalDocumentChunker:
    """
    Intelligent chunking for legal documents that preserves hierarchical structure
    """
    
    def __init__(self, max_chunk_tokens=512, overlap_tokens=50):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        
    def extract_act_name(self, text: str) -> str:
        """Extract act name from text"""
        match = re.search(r'THE\s+([A-Z\s,]+ACT,?\s*\d{4})', text, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown Act"
    
    def extract_section_number(self, text: str) -> str:
        """Extract section number"""
        match = re.search(r'(?:Section|SECTION)\s+(\d+[A-Z]?)', text, re.IGNORECASE)
        return match.group(1) if match else "Unknown"
    
    def split_by_sections(self, text: str) -> List[Dict]:
        """Split document by sections while preserving context"""
        chunks = []
        chunk_id = 0
        
        # Split by major acts first
        act_pattern = r'(?=THE\s+[A-Z\s,]+ACT,?\s*\d{4})'
        acts = re.split(act_pattern, text, flags=re.IGNORECASE)
        
        for act_text in acts:
            if len(act_text.strip()) < 50:
                continue
                
            act_name = self.extract_act_name(act_text)
            
            # Split by sections within each act
            section_pattern = r'(?=(?:^|\n)(?:Section|\d+\.)\s+\d+[A-Z]?\.?)'
            sections = re.split(section_pattern, act_text, flags=re.MULTILINE)
            
            for section_text in sections:
                section_text = section_text.strip()
                if len(section_text) < 30:
                    continue
                
                section_num = self.extract_section_number(section_text)
                
                # Check if section is too long
                words = section_text.split()
                
                if len(words) <= self.max_chunk_tokens:
                    # Keep entire section as one chunk
                    chunks.append({
                        'id': chunk_id,
                        'text': section_text,
                        'metadata': {
                            'act': act_name,
                            'section': section_num,
                            'type': 'full_section',
                            'word_count': len(words)
                        }
                    })
                    chunk_id += 1
                else:
                    # Split long sections with overlap
                    sub_chunks = self._split_with_overlap(
                        section_text, 
                        self.max_chunk_tokens,
                        self.overlap_tokens
                    )
                    
                    for idx, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            'id': chunk_id,
                            'text': sub_chunk,
                            'metadata': {
                                'act': act_name,
                                'section': section_num,
                                'type': 'section_part',
                                'part': idx + 1,
                                'total_parts': len(sub_chunks),
                                'word_count': len(sub_chunk.split())
                            }
                        })
                        chunk_id += 1
        
        return chunks
    
    def _split_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks by tokens"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            end = min(i + chunk_size, len(words))
            chunk_words = words[i:end]
            chunks.append(' '.join(chunk_words))
            
            if end >= len(words):
                break
            
            i += (chunk_size - overlap)
        
        return chunks
    
    def chunk_document(self, text: str) -> List[Dict]:
        """Main chunking method"""
        logger.info("Starting semantic chunking...")
        chunks = self.split_by_sections(text)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        logger.info(f"Average chunk size: {sum(c['metadata']['word_count'] for c in chunks) / len(chunks):.1f} words")
        
        return chunks


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    input_file = data_dir / 'regulatory-chatbot-api.txt'
    output_pkl = data_dir / 'chunks_v2.pkl'
    output_jsonl = data_dir / 'chunks_v2.jsonl'
    
    # Load document
    logger.info(f"Loading document from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    logger.info(f"Document size: {len(text)} characters, {len(text.split())} words")
    
    # Create chunker with token-based limits (not word-based)
    chunker = LegalDocumentChunker(
        max_chunk_tokens=512,  # More reasonable for embeddings
        overlap_tokens=50
    )
    
    # Generate chunks
    chunks = chunker.chunk_document(text)
    
    # Save as pickle
    logger.info(f"Saving chunks to {output_pkl}")
    with open(output_pkl, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save as JSONL
    logger.info(f"Saving chunks to {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    # Statistics
    logger.info("\n" + "="*60)
    logger.info("CHUNKING STATISTICS")
    logger.info("="*60)
    logger.info(f"Total chunks: {len(chunks)}")
    
    full_sections = [c for c in chunks if c['metadata']['type'] == 'full_section']
    section_parts = [c for c in chunks if c['metadata']['type'] == 'section_part']
    
    logger.info(f"Full sections: {len(full_sections)}")
    logger.info(f"Split sections: {len(section_parts)}")
    
    word_counts = [c['metadata']['word_count'] for c in chunks]
    logger.info(f"Chunk size - Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)/len(word_counts):.1f}")
    
    # Show sample chunks
    logger.info("\nSample chunks:")
    for i in range(min(3, len(chunks))):
        chunk = chunks[i]
        logger.info(f"\nChunk {i}:")
        logger.info(f"  Act: {chunk['metadata']['act']}")
        logger.info(f"  Section: {chunk['metadata']['section']}")
        logger.info(f"  Type: {chunk['metadata']['type']}")
        logger.info(f"  Text preview: {chunk['text'][:200]}...")
    
    logger.info("\n✓ Semantic chunking complete!")


if __name__ == '__main__':
    main()

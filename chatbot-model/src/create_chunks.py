"""
Semantic Chunking for Mining Laws
Creates chunks from mining laws document and saves to data folder
"""

import re
import json
from pathlib import Path
from typing import List, Dict


class SemanticChunker:
    def __init__(self):
        self.data_dir = Path("data")
        self.input_file = self.data_dir / "mining_laws.txt"
        self.output_file = self.data_dir / "mining_laws_chunks.json"
        self.chunks = []
        
    def read_document(self) -> str:
        """Read the mining laws document"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def identify_act_sections(self, text: str) -> List[Dict]:
        """
        Identify different acts and their sections
        Returns list of sections with metadata
        """
        sections = []
        
        # Split by major acts
        acts = self._split_by_acts(text)
        
        for act_name, act_content in acts:
            # Split each act into sections
            act_sections = self._split_by_sections(act_content, act_name)
            sections.extend(act_sections)
        
        return sections
    
    def _split_by_acts(self, text: str) -> List[tuple]:
        """Split document into different acts"""
        acts = []
        
        # Pattern to identify major acts (all caps title)
        act_pattern = r'THE\s+([A-Z\s,]+?ACT,\s+\d{4})'
        
        matches = list(re.finditer(act_pattern, text))
        
        for i, match in enumerate(matches):
            act_name = match.group(0).strip()
            start_pos = match.start()
            
            # Find end position (start of next act or end of document)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            act_content = text[start_pos:end_pos]
            acts.append((act_name, act_content))
        
        return acts
    
    def _split_by_sections(self, act_content: str, act_name: str) -> List[Dict]:
        """Split act content into sections"""
        sections = []
        
        # Pattern for section numbers: "1.", "2.", "40.", etc.
        section_pattern = r'^(\d+[A-Z]?)\.\s+([^\n]+)'
        
        lines = act_content.split('\n')
        current_section_number = None
        current_section_title = None
        current_text = []
        
        for line in lines:
            # Check if line starts a new section
            match = re.match(section_pattern, line.strip())
            
            if match:
                # Save previous section if exists
                if current_section_number and current_text:
                    section_text = '\n'.join(current_text).strip()
                    if section_text and len(section_text.split()) > 20:
                        chunk = self._create_chunk(
                            section_text,
                            act_name,
                            current_section_number,
                            current_section_title
                        )
                        if isinstance(chunk, list):
                            sections.extend(chunk)
                        else:
                            sections.append(chunk)
                
                # Start new section
                current_section_number = match.group(1)
                current_section_title = match.group(2).strip()
                current_text = [line]
            else:
                if current_section_number:
                    current_text.append(line)
        
        # Save last section
        if current_section_number and current_text:
            section_text = '\n'.join(current_text).strip()
            if section_text and len(section_text.split()) > 20:
                chunk = self._create_chunk(
                    section_text,
                    act_name,
                    current_section_number,
                    current_section_title
                )
                if isinstance(chunk, list):
                    sections.extend(chunk)
                else:
                    sections.append(chunk)
        
        return sections
    
    def _create_chunk(self, text: str, act_name: str, section_number: str, 
                     section_title: str) -> Dict:
        """Create a chunk dictionary with metadata"""
        words = text.split()
        word_count = len(words)
        
        # If section is too long, split it further
        if word_count > 500:
            return self._split_long_section(text, act_name, section_number, section_title)
        
        return {
            "text": text,
            "metadata": {
                "act": act_name,
                "section": section_number,
                "section_title": section_title,
                "word_count": word_count,
                "source": "mining_laws.txt"
            }
        }
    
    def _split_long_section(self, text: str, act_name: str, section_number: str, 
                           section_title: str) -> List[Dict]:
        """Split long sections into smaller chunks with overlap"""
        words = text.split()
        chunks = []
        chunk_size = 400
        overlap = 50
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            part_num = i // (chunk_size - overlap) + 1
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "act": act_name,
                    "section": f"{section_number} (part {part_num})",
                    "section_title": section_title,
                    "word_count": len(chunk_words),
                    "source": "mining_laws.txt"
                }
            })
        
        return chunks
    
    def create_chunks(self) -> List[Dict]:
        """Main method to create chunks from document"""
        print("Reading document...")
        text = self.read_document()
        
        print("Identifying sections and creating chunks...")
        sections = self.identify_act_sections(text)
        
        # Add chunk IDs
        for idx, section in enumerate(sections, 1):
            section['chunk_id'] = idx
        
        self.chunks = sections
        print(f"Created {len(self.chunks)} chunks")
        
        return self.chunks
    
    def save_chunks(self):
        """Save chunks to JSON file"""
        output_data = {
            "total_chunks": len(self.chunks),
            "chunks": self.chunks
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Chunks saved to {self.output_file}")
    
    def print_statistics(self):
        """Print chunk statistics"""
        if not self.chunks:
            return
        
        word_counts = [c['metadata']['word_count'] for c in self.chunks]
        acts = set(c['metadata']['act'] for c in self.chunks)
        
        print(f"\n{'='*80}")
        print("CHUNKING STATISTICS")
        print(f"{'='*80}")
        print(f"Total chunks: {len(self.chunks)}")
        print(f"Acts covered: {len(acts)}")
        for act in acts:
            count = sum(1 for c in self.chunks if c['metadata']['act'] == act)
            print(f"  - {act}: {count} chunks")
        print(f"\nWord count per chunk:")
        print(f"  - Average: {sum(word_counts)/len(word_counts):.1f} words")
        print(f"  - Min: {min(word_counts)} words")
        print(f"  - Max: {max(word_counts)} words")
        print(f"{'='*80}")
    
    def print_sample_chunks(self, n=3):
        """Print sample chunks for verification"""
        print(f"\n{'='*80}")
        print(f"SAMPLE CHUNKS (first {n}):")
        print(f"{'='*80}\n")
        
        for chunk in self.chunks[:n]:
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Act: {chunk['metadata']['act']}")
            print(f"Section: {chunk['metadata']['section']}")
            print(f"Title: {chunk['metadata']['section_title']}")
            print(f"Word Count: {chunk['metadata']['word_count']}")
            print(f"Text Preview: {chunk['text'][:200]}...")
            print(f"{'-'*80}\n")


def main():
    print("="*80)
    print("SEMANTIC CHUNKING FOR MINING LAWS")
    print("="*80 + "\n")
    
    chunker = SemanticChunker()
    
    # Create chunks
    chunker.create_chunks()
    
    # Save to file
    chunker.save_chunks()
    
    # Print statistics and samples
    chunker.print_statistics()
    chunker.print_sample_chunks(n=5)
    
    print("\n✓ Chunking completed successfully!\n")


if __name__ == "__main__":
    main()

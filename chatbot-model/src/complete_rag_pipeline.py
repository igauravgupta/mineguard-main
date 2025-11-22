"""
Complete RAG Pipeline: End-to-End Mining Laws Chatbot
Integrates all phases: Chunking → Embedding → Retrieval → Answer Generation

STEP 14 COMPLIANT: Full orchestration with post-processing
Pipeline:
1. Receive user query
2. Run hybrid retrieval (FAISS + BM25 + RRF)
3. Select best chunk(s) (top-k)
4. Construct prompt (system role + context + query)
5. Send to LLM and receive answer
6. Post-process (clean, cite, validate)
7. Return structured response
"""

import os
import sys
import re
from typing import Dict, List, Optional

# Import components from previous phases
from src.retrieval_system import RetrievalSystem
from src.answer_generator import AnswerGenerator


class AnswerPostProcessor:
    """
    Post-processing for LLM-generated answers (Step 14 requirement)
    - Clean formatting
    - Add/validate citations
    - Validate length
    - Extract structured components
    """
    
    @staticmethod
    def clean_formatting(text: str) -> str:
        """Remove extra whitespace and normalize formatting"""
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing whitespace on lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        # Normalize spaces
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extract section/act citations from answer"""
        citations = []
        
        # Pattern: "Section X" or "Section X of Act Y"
        section_pattern = r'Section\s+\d+[A-Za-z]*(?:\s+of\s+[^.]+)?'
        citations.extend(re.findall(section_pattern, text, re.IGNORECASE))
        
        # Pattern: Act names
        act_pattern = r'(?:Mines|Explosives)\s+Act\s+\d{4}'
        citations.extend(re.findall(act_pattern, text, re.IGNORECASE))
        
        return list(set(citations))  # Remove duplicates
    
    @staticmethod
    def validate_answer(answer: str, min_length: int = 100) -> Dict[str, any]:
        """
        Validate answer quality
        
        Returns:
            Dict with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'word_count': len(answer.split()),
            'char_count': len(answer)
        }
        
        # Check minimum length
        if len(answer) < min_length:
            validation['is_valid'] = False
            validation['issues'].append(f"Answer too short ({len(answer)} chars, min: {min_length})")
        
        # Check if answer is substantive (not just "I don't know")
        no_answer_phrases = [
            "i don't know",
            "no information",
            "cannot answer",
            "not provided",
            "insufficient context"
        ]
        if any(phrase in answer.lower() for phrase in no_answer_phrases):
            validation['issues'].append("Answer indicates insufficient information")
        
        # Check for citations
        citations = AnswerPostProcessor.extract_citations(answer)
        if not citations:
            validation['issues'].append("No legal citations found")
        
        return validation
    
    @staticmethod
    def add_chunk_citations(answer: str, chunks: List[Dict]) -> str:
        """
        Ensure answer includes proper citations from chunks
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks used
        
        Returns:
            Answer with enhanced citations
        """
        # Extract existing citations
        existing = AnswerPostProcessor.extract_citations(answer)
        
        # If no citations, add them from chunks
        if not existing and chunks:
            footer = "\n\n**Sources:**\n"
            for i, chunk in enumerate(chunks[:3], 1):  # Top 3 chunks
                meta = chunk['metadata']
                section = meta.get('section', 'Unknown')
                act = meta.get('act', 'Unknown Act')
                footer += f"{i}. {act}, {section}\n"
            
            answer += footer
        
        return answer
    
    @staticmethod
    def structure_response(
        answer: str,
        chunks: List[Dict],
        query: str,
        tokens_used: Dict = None
    ) -> Dict:
        """
        Create Step 14 compliant structured response
        
        Returns:
            {
                "answer": "...",
                "source": "Section X, Act Y",
                "sources": [...],
                "chunk_used": {...},
                "chunks_used": [...],
                "validation": {...},
                "query": "...",
                "tokens_used": {...}
            }
        """
        # Clean answer
        clean_answer = AnswerPostProcessor.clean_formatting(answer)
        
        # Validate
        validation = AnswerPostProcessor.validate_answer(clean_answer)
        
        # Extract citations
        citations = AnswerPostProcessor.extract_citations(clean_answer)
        
        # Primary source (from top chunk)
        primary_source = "Unknown"
        if chunks:
            meta = chunks[0]['metadata']
            section = meta.get('section', 'Unknown')
            act = meta.get('act', 'Unknown Act')
            primary_source = f"{section}, {act}"
        
        # Build structured response
        response = {
            "answer": clean_answer,
            "source": primary_source,  # Primary citation (Step 14 format)
            "sources": citations,  # All extracted citations
            "chunk_used": chunks[0] if chunks else None,  # Top chunk (Step 14 format)
            "chunks_used": chunks,  # All chunks used
            "validation": validation,
            "query": query,
            "tokens_used": tokens_used or {}
        }
        
        return response


class MiningLawsChatbot:
    """
    Complete RAG-based Chatbot for Mining Laws
    
    STEP 14 COMPLIANT PIPELINE:
    1. Receive user query
    2. Run hybrid retrieval (FAISS + BM25 + RRF)
    3. Select best chunk(s) (top-k)
    4. Construct prompt (system role + context + query)
    5. Send to LLM and receive answer
    6. Post-process (clean, cite, validate)
    7. Return structured response
    """
    
    def __init__(
        self,
        index_path: str = "models/mining_laws_finetuned.index",
        chunks_path: str = "data/mining_laws_chunks.json",
        model_path: str = "models/bi_encoder_finetuned",
        groq_api_key: Optional[str] = None,
        retrieval_method: str = "hybrid",
        top_k: int = 5,
        min_answer_length: int = 100
    ):
        """
        Initialize complete RAG pipeline
        
        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks JSON
            model_path: Path to sentence transformer model
            groq_api_key: Groq API key for answer generation
            retrieval_method: 'dense', 'sparse', or 'hybrid'
            top_k: Number of chunks to retrieve
            min_answer_length: Minimum answer length for validation
        """
        print("="*70)
        print("Initializing Mining Laws Chatbot (Step 14 Compliant Pipeline)")
        print("="*70)
        
        # Determine whether to use fine-tuned or base model
        use_finetuned = os.path.exists(index_path)
        
        # Fallback to base index if finetuned doesn't exist
        if not use_finetuned:
            print(f"⚠ Fine-tuned index not found: {index_path}")
            index_path = "models/mining_laws_base.index"
            model_path = "sentence-transformers/all-mpnet-base-v2"
            print(f"  Using base index: {index_path}")
        
        # Initialize retrieval system
        print("\n[1/3] Loading Retrieval System...")
        self.retrieval = RetrievalSystem(use_finetuned=use_finetuned)
        self.retrieval_method = retrieval_method
        self.top_k = top_k
        
        # Initialize answer generator
        print("\n[2/3] Loading Answer Generator...")
        self.generator = AnswerGenerator(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024
        )
        
        # Initialize post-processor
        print("\n[3/3] Initializing Post-Processor...")
        self.post_processor = AnswerPostProcessor()
        self.min_answer_length = min_answer_length
        
        print("\n" + "="*70)
        print("✓ Chatbot Ready (Step 14 Compliant)!")
        print(f"  Retrieval: {retrieval_method.upper()} | Top-K: {top_k}")
        print(f"  Model: Groq Llama 3.1 70B")
        print(f"  Post-processing: Enabled (citations, validation)")
        print("="*70 + "\n")
    
    def ask(self, query: str, verbose: bool = True) -> Dict:
        """
        STEP 14 COMPLIANT PIPELINE
        
        Complete RAG pipeline: Query → Retrieve → Generate → Post-process
        
        Args:
            query: User's question
            verbose: Print intermediate steps
        
        Returns:
            Structured response dict (Step 14 format):
            {
                "answer": str,
                "source": str (primary citation),
                "sources": List[str] (all citations),
                "chunk_used": Dict (top chunk),
                "chunks_used": List[Dict] (all chunks),
                "validation": Dict,
                "query": str,
                "tokens_used": Dict
            }
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"USER QUERY: {query}")
            print(f"{'='*70}\n")
        
        # STEP 14.1: Receive user query ✓
        
        # STEP 14.2: Run hybrid retrieval (FAISS + BM25 + RRF)
        if verbose:
            print(f"[Step 1/4] Hybrid Retrieval ({self.retrieval_method})...")
        
        retrieved_chunks = self.retrieval.search(
            query=query,
            method=self.retrieval_method,
            top_k=self.top_k
        )
        
        if verbose:
            print(f"✓ Retrieved {len(retrieved_chunks)} chunks\n")
            print("Top 3 chunks:")
            for i, chunk in enumerate(retrieved_chunks[:3], 1):
                metadata = chunk['metadata']
                print(f"  {i}. {metadata.get('act', 'Unknown')} - {metadata.get('section', 'Unknown')}")
                print(f"     Score: {chunk['score']:.3f}")
            print()
        
        # STEP 14.3: Select best chunks ✓ (already have top-k)
        
        # STEP 14.4: Construct prompt (done in AnswerGenerator)
        # STEP 14.5: Send to LLM
        if verbose:
            print(f"[Step 2/4] Generating answer with LLM...")
        
        llm_result = self.generator.generate_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            verbose=False
        )
        
        if verbose:
            print(f"✓ Answer generated ({llm_result['tokens_used']['total_tokens']} tokens)\n")
        
        # STEP 14.6: Post-process (clean, cite, validate)
        if verbose:
            print(f"[Step 3/4] Post-processing answer...")
        
        # Clean formatting
        clean_answer = self.post_processor.clean_formatting(llm_result['answer'])
        
        # Ensure citations
        cited_answer = self.post_processor.add_chunk_citations(
            clean_answer,
            retrieved_chunks
        )
        
        # Validate
        validation = self.post_processor.validate_answer(
            cited_answer,
            min_length=self.min_answer_length
        )
        
        if verbose:
            print(f"✓ Post-processing complete")
            print(f"  Word count: {validation['word_count']}")
            print(f"  Citations found: {len(self.post_processor.extract_citations(cited_answer))}")
            print(f"  Valid: {validation['is_valid']}")
            if validation['issues']:
                print(f"  Issues: {', '.join(validation['issues'])}")
            print()
        
        # STEP 14.7: Return structured response
        if verbose:
            print(f"[Step 4/4] Creating structured response...")
        
        response = self.post_processor.structure_response(
            answer=cited_answer,
            chunks=retrieved_chunks,
            query=query,
            tokens_used=llm_result['tokens_used']
        )
        
        # Add retrieval metadata
        response['retrieval_method'] = self.retrieval_method
        response['top_k'] = self.top_k
        
        if verbose:
            print(f"✓ Response structured (Step 14 format)\n")
        
        return response
    
    def print_response(self, response: Dict):
        """
        Pretty print Step 14 compliant response
        """
        print("\n" + "="*70)
        print("ANSWER")
        print("="*70)
        print(response['answer'])
        
        print("\n" + "-"*70)
        print("METADATA")
        print("-"*70)
        print(f"Primary Source: {response['source']}")
        
        if response.get('sources'):
            print(f"\nAll Citations:")
            for i, cite in enumerate(response['sources'], 1):
                print(f"  {i}. {cite}")
        
        print(f"\nValidation:")
        val = response['validation']
        print(f"  Status: {'✓ Valid' if val['is_valid'] else '✗ Issues found'}")
        print(f"  Word count: {val['word_count']}")
        if val['issues']:
            print(f"  Issues: {', '.join(val['issues'])}")
        
        print(f"\nTokens used: {response['tokens_used']['total_tokens']}")
        print(f"Retrieval method: {response['retrieval_method']}")
        print("="*70)
    
    def interactive(self):
        """Interactive chat mode"""
        print("\n" + "="*70)
        print("INTERACTIVE MODE - Mining Laws Chatbot (Step 14 Compliant)")
        print("="*70)
        print("Ask questions about Indian mining and explosives laws.")
        print("Type 'quit', 'exit', or 'q' to stop.\n")
        
        while True:
            try:
                # Get user input
                query = input("\n🔍 Your question: ").strip()
                
                # Check for exit
                if query.lower() in ['quit', 'exit', 'q', '']:
                    print("\n👋 Goodbye!\n")
                    break
                
                # Get answer (Step 14 pipeline)
                response = self.ask(query, verbose=True)
                
                # Print formatted response
                self.print_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
    
    def batch_query(self, queries: List[str], save_results: bool = True):
        """
        Process multiple queries and save Step 14 compliant results
        
        Args:
            queries: List of questions
            save_results: Save to JSON file
        """
        results = []
        
        print(f"\n{'='*70}")
        print(f"Processing {len(queries)} queries (Step 14 Pipeline)...")
        print(f"{'='*70}\n")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[Query {i}/{len(queries)}]")
            response = self.ask(query, verbose=False)
            results.append(response)
            
            # Print compact summary
            print(f"✓ Query: {query[:60]}...")
            print(f"  Answer length: {response['validation']['word_count']} words")
            print(f"  Valid: {response['validation']['is_valid']}")
            print(f"  Citations: {len(response['sources'])}")
            print(f"  Tokens: {response['tokens_used']['total_tokens']}")
        
        # Save to file
        if save_results:
            import json
            output_file = "batch_results_step14.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to: {output_file}")
        
        return results


def demo():
    """Demo with sample queries showing Step 14 compliance"""
    print("\n" + "="*70)
    print("DEMO: Step 14 Compliant RAG Pipeline")
    print("="*70)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠ GROQ_API_KEY not found!")
        print("\nSetup required:")
        print("1. Get free API key from: https://console.groq.com")
        print("2. Set environment variable:")
        print("   export GROQ_API_KEY='your-api-key-here'")
        print("3. Run again\n")
        return
    
    # Initialize chatbot
    try:
        chatbot = MiningLawsChatbot(
            retrieval_method="hybrid",
            top_k=5,
            min_answer_length=100
        )
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        print("\nMake sure you've run:")
        print("  python build_base_index.py")
        return
    
    # Sample queries
    sample_queries = [
        "What licenses are required to manufacture explosives in India?",
        "What are the safety requirements for storing explosives?",
        "Can someone appeal if their explosives license is denied?"
    ]
    
    print("\n" + "="*70)
    print("Testing Step 14 Pipeline with sample queries...")
    print("="*70)
    
    for query in sample_queries:
        response = chatbot.ask(query, verbose=True)
        chatbot.print_response(response)
        print("\n" + "-"*70 + "\n")
    
    print("✓ Demo completed!")
    print("\nStep 14 Pipeline Features Demonstrated:")
    print("  ✓ Hybrid retrieval (FAISS + BM25 + RRF)")
    print("  ✓ LLM answer generation")
    print("  ✓ Post-processing (clean, cite, validate)")
    print("  ✓ Structured response format")
    print("\nTo start interactive mode:")
    print("  python complete_rag_pipeline.py --interactive")


if __name__ == "__main__":
    # Check if user wants demo or interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Interactive mode
        if not os.getenv("GROQ_API_KEY"):
            print("Error: GROQ_API_KEY environment variable not set")
            sys.exit(1)
        
        chatbot = MiningLawsChatbot()
        chatbot.interactive()
    else:
        # Demo mode
        demo()
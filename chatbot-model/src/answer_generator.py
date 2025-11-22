"""
Phase 4: Answer Generation (Steps 12-13)
Uses Groq API with Llama 3.1 70B for high-quality legal answer generation
"""

import os
import json
from typing import List, Dict, Optional
from groq import Groq


class AnswerGenerator:
    """
    Step 12: Answer Generation Model (Groq Llama 3.1 70B)
    Step 13: Prompt Engineering for Legal Q&A
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        """
        Initialize Answer Generator with Groq API
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env variable)
            model: Model to use (llama-3.3-70b-versatile, llama-3.1-8b-instant, etc.)
            temperature: 0.0-1.0 (lower = more focused, higher = more creative)
            max_tokens: Maximum response length
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Either:\n"
                "1. Pass api_key parameter, or\n"
                "2. Set GROQ_API_KEY environment variable\n"
                "Get free key from: https://console.groq.com"
            )
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"✓ Initialized AnswerGenerator with model: {model}")
    
    def _format_retrieved_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context for the prompt
        
        Args:
            retrieved_chunks: List of dicts with 'text', 'metadata', 'score'
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Extract section info from metadata
            act = metadata.get('act', 'Unknown Act')
            section = metadata.get('section', 'Unknown Section')
            
            context_parts.append(
                f"[Context {i}] {act} - {section}\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def _create_legal_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        Step 13: Design Prompt Engineering for Legal Q&A
        
        Creates a structured prompt with role, context, and instructions
        
        Args:
            query: User's question
            context: Retrieved legal text chunks
        
        Returns:
            Messages list for Groq API
        """
        system_prompt = """You are an expert legal assistant specializing in Indian mining and explosives regulations. Your role is to provide accurate, well-cited answers based on the Explosives Act 1884 and Mines Act 1952.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided context - do not use external knowledge
2. Always cite specific section numbers and act names
3. Structure your answer clearly with proper formatting
4. Include relevant details, requirements, and exceptions
5. If the context doesn't contain enough information, say so clearly
6. Aim for comprehensive answers (minimum 150 words when possible)

ANSWER FORMAT:
Use this structured format for every response:

**Direct Answer:**
[Provide the main answer to the question]

**Legal Basis:**
[Cite specific sections and acts that support the answer]

**Key Requirements:**
[List important requirements, procedures, or conditions]

**Exceptions (if any):**
[Mention any exceptions, special cases, or limitations]

Remember: Accuracy and proper citation are paramount in legal matters."""

        user_prompt = f"""Based on the following legal context, answer the user's question:

CONTEXT:
{context}

USER QUESTION:
{query}

Provide a comprehensive, well-cited answer following the required format."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Generate answer using Groq API
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved chunks from retrieval system
            verbose: Print generation details
        
        Returns:
            Dict with 'answer', 'model', 'tokens_used', 'chunks_used'
        """
        # Format context
        context = self._format_retrieved_context(retrieved_chunks)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generating answer for: {query}")
            print(f"Using {len(retrieved_chunks)} retrieved chunks")
            print(f"{'='*60}\n")
        
        # Create prompt
        messages = self._create_legal_prompt(query, context)
        
        # Call Groq API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Get token usage
            tokens_used = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            if verbose:
                print(f"✓ Answer generated successfully")
                print(f"  Tokens used: {tokens_used['total_tokens']} "
                      f"(prompt: {tokens_used['prompt_tokens']}, "
                      f"completion: {tokens_used['completion_tokens']})")
            
            return {
                'answer': answer,
                'query': query,
                'model': self.model,
                'tokens_used': tokens_used,
                'chunks_used': len(retrieved_chunks),
                'chunks': retrieved_chunks  # Include for reference
            }
            
        except Exception as e:
            print(f"✗ Error generating answer: {str(e)}")
            raise
    
    def print_answer(self, result: Dict):
        """Pretty print the generated answer"""
        print(f"\n{'='*70}")
        print(f"QUESTION: {result['query']}")
        print(f"{'='*70}\n")
        print(result['answer'])
        print(f"\n{'='*70}")
        print(f"Model: {result['model']} | Tokens: {result['tokens_used']['total_tokens']} | "
              f"Chunks: {result['chunks_used']}")
        print(f"{'='*70}\n")


def demo():
    """
    Demo of answer generation system
    Requires: GROQ_API_KEY environment variable
    """
    print("="*70)
    print("Phase 4: Answer Generation Demo (Steps 12-13)")
    print("="*70)
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠ GROQ_API_KEY not found!")
        print("\nTo use this demo:")
        print("1. Get free API key from: https://console.groq.com")
        print("2. Set environment variable:")
        print("   export GROQ_API_KEY='your-api-key-here'")
        print("3. Run this script again\n")
        return
    
    # Initialize generator
    generator = AnswerGenerator(
        model="llama-3.3-70b-versatile",  # Best quality
        temperature=0.3,  # Focused, deterministic answers
        max_tokens=1024
    )
    
    # Example: Simulate retrieved chunks
    sample_chunks = [
        {
            'text': 'No person shall manufacture, possess for sale, sell, transport, import or use any explosive except under and in accordance with a licence granted in this behalf and in accordance with the provisions of this Act and the rules made thereunder.',
            'metadata': {
                'act': 'Explosives Act 1884',
                'section': 'Section 4 - Prohibition of manufacture, possession, etc., without license'
            },
            'score': 0.89
        },
        {
            'text': 'The licensing authority may grant licenses for the manufacture, possession, sale, transport, import or use of explosives subject to such restrictions and conditions as it thinks fit to impose. Every such license shall be in such form and shall be granted on payment of such fee as may be prescribed.',
            'metadata': {
                'act': 'Explosives Act 1884',
                'section': 'Section 5 - Grant of licenses'
            },
            'score': 0.85
        },
        {
            'text': 'Any person aggrieved by an order made by a licensing authority refusing to grant or renew a license or revoking or suspending a license may appeal to the prescribed authority within such period as may be prescribed.',
            'metadata': {
                'act': 'Explosives Act 1884',
                'section': 'Section 6 - Appeals'
            },
            'score': 0.78
        }
    ]
    
    # Test query
    test_query = "What is required to manufacture or sell explosives in India?"
    
    # Generate answer
    result = generator.generate_answer(
        query=test_query,
        retrieved_chunks=sample_chunks,
        verbose=True
    )
    
    # Print result
    generator.print_answer(result)
    
    print("\n✓ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Integrate with retrieval_system.py for end-to-end RAG")
    print("2. Test with real queries from your domain")
    print("3. Adjust temperature (0.2-0.4) for optimal balance")


if __name__ == "__main__":
    demo()

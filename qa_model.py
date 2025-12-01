"""
Question Answering Model Handler
Uses transformers library with RoBERTa-base-SQuAD2 for answer extraction
"""

from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class QAModel:
    """
    Question Answering model wrapper using HuggingFace transformers
    """
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize QA model
        
        Args:
            model_name: Name of the QA model from HuggingFace
        """
        print(f"Loading QA model: {model_name}")
        
        try:
            # Load model and tokenizer
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Create pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print(f"QA model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        except Exception as e:
            print(f"Error loading QA model: {e}")
            raise
    
    def answer_question(
        self,
        question: str,
        context: str,
        max_answer_length: int = 150,
        min_score: float = 0.01
    ) -> Tuple[Optional[str], float]:
        """
        Answer a question based on given context
        
        Args:
            question: User question
            context: Context text to extract answer from
            max_answer_length: Maximum length of answer
            min_score: Minimum confidence score threshold
        
        Returns:
            (answer, confidence_score) or (None, 0.0) if no answer found
        """
        if not context or not question:
            return None, 0.0
        
        try:
            # Ensure inputs are strings
            question = str(question).strip()
            context = str(context).strip()
            
            if not question or not context:
                return None, 0.0
            
            # Truncate context if too long (BERT has 512 token limit)
            max_context_length = 4000  # Characters, not tokens
            if len(context) > max_context_length:
                context = context[:max_context_length]
            
            # Get answer from pipeline
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=max_answer_length,
                handle_impossible_answer=True
            )
            
            answer = result.get('answer', '').strip()
            score = result.get('score', 0.0)
            
            # Filter low-confidence answers
            if score < min_score or not answer:
                return None, 0.0
            
            return answer, float(score)
        
        except Exception as e:
            print(f"Error in QA model: {e}")
            return None, 0.0
    
    def batch_answer(
        self,
        question: str,
        contexts: list,
        max_answer_length: int = 150
    ) -> list:
        """
        Answer a question using multiple contexts and return all results
        
        Args:
            question: User question
            contexts: List of context texts
            max_answer_length: Maximum length of answer
        
        Returns:
            List of (answer, score, context_index) tuples
        """
        results = []
        
        for idx, context in enumerate(contexts):
            answer, score = self.answer_question(
                question,
                context,
                max_answer_length=max_answer_length
            )
            
            if answer:
                results.append((answer, score, idx))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


class SmartQAHandler:
    """
    Intelligent QA handler that combines semantic search with QA model
    """
    
    def __init__(self, semantic_engine, qa_model: QAModel):
        """
        Initialize smart QA handler
        
        Args:
            semantic_engine: SemanticSearchEngine instance
            qa_model: QAModel instance
        """
        self.semantic_engine = semantic_engine
        self.qa_model = qa_model
    
    def answer(
        self,
        question: str,
        fetch_new_data: bool = False,
        sources: list = None
    ) -> Dict:
        """
        Answer a question using semantic search + QA model
        
        Args:
            question: User question
            fetch_new_data: Whether to fetch new data from sources
            sources: List of sources to use if fetching new data
        
        Returns:
            Dict with 'answer', 'score', 'sources', 'context'
        """
        
        # If requested, fetch new data for this question
        if fetch_new_data:
            from source_fetchers import SourceAggregator
            aggregator = SourceAggregator()
            
            if sources is None:
                sources = ['wikipedia', 'stackoverflow']
            
            print(f"Fetching fresh data from: {sources}")
            new_docs = aggregator.search_all(question, sources=sources)
            
            if new_docs:
                self.semantic_engine.add_documents(new_docs, chunk=True)
                self.semantic_engine.save()
        
        # Get relevant context using semantic search
        context, source_docs = self.semantic_engine.get_context_for_qa(
            question,
            top_k=3
        )
        
        print(f"DEBUG: Found {len(source_docs)} relevant documents")
        
        if not context:
            return {
                'answer': None,
                'score': 0.0,
                'sources': [],
                'context': '',
                'message': 'I could not find reliable information on this topic in my sources.'
            }
        
        print(f"DEBUG: Context length: {len(context)} chars")
        
        # Extract answer using QA model
        answer, score = self.qa_model.answer_question(
            question,
            context,
            max_answer_length=200,
            min_score=0.01
        )
        
        print(f"DEBUG: QA model returned: answer={answer is not None}, score={score:.3f}")
        
        # If no answer found, try with individual contexts
        if not answer and source_docs:
            print("Trying individual contexts...")
            for i, doc in enumerate(source_docs[:2]):  # Try top 2
                print(f"  Trying document {i+1}: {doc.get('title', 'Unknown')[:50]}...")
                answer, score = self.qa_model.answer_question(
                    question,
                    doc['text'],
                    max_answer_length=200,
                    min_score=0.01
                )
                if answer:
                    print(f"  Found answer in document {i+1}!")
                    break
        
        # Format sources
        sources_list = []
        for doc in source_docs:
            sources_list.append({
                'source': doc['source'],
                'url': doc.get('url', ''),
                'relevance': doc.get('score', 0.0)
            })
        
        if answer:
            return {
                'answer': answer,
                'score': score,
                'sources': sources_list,
                'context': context[:500] + '...' if len(context) > 500 else context,
                'message': None
            }
        else:
            return {
                'answer': None,
                'score': 0.0,
                'sources': sources_list,
                'context': context[:500] + '...' if len(context) > 500 else context,
                'message': 'I could not find a clear answer in my sources, but I found some relevant information.'
            }
    
    def format_response(self, result: Dict) -> str:
        """
        Format the QA result into a user-friendly response
        
        Args:
            result: Result dict from answer()
        
        Returns:
            Formatted response string
        """
        if result['answer']:
            response = result['answer']
            
            # Add sources
            if result['sources']:
                response += "\n\n[Sources: "
                source_names = [s['source'] for s in result['sources'][:3]]
                response += ", ".join(source_names)
                response += "]"
            
            return response
        else:
            if result['message']:
                return result['message']
            else:
                return "I could not find reliable information on this topic in my sources."


def initialize_qa_system():
    """
    Initialize the complete QA system with semantic search and QA model
    
    Returns:
        SmartQAHandler instance
    """
    from semantic_engine import SemanticSearchEngine
    
    print("Initializing QA system...")
    
    # Initialize semantic search engine
    semantic_engine = SemanticSearchEngine(model_name="all-MiniLM-L6-v2")
    
    # Initialize QA model
    qa_model = QAModel(model_name="deepset/roberta-base-squad2")
    
    # Create smart handler
    handler = SmartQAHandler(semantic_engine, qa_model)
    
    print("QA system initialized successfully")
    
    return handler

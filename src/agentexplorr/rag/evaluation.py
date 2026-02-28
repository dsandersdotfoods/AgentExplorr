"""
RAG Evaluation Metrics — Measuring Retrieval & Generation Quality
===================================================================

WHY EVALUATE RAG?
  Building a RAG system is easy.  Building a GOOD RAG system is hard.
  Without evaluation, you're flying blind:
    - Are you retrieving the RIGHT chunks?
    - Is the LLM actually USING the retrieved context?
    - Is the answer FAITHFUL to the sources (not hallucinated)?
    - Is the answer RELEVANT to the user's question?

  Evaluation metrics give you quantitative signals to guide improvements
  to your chunking strategy, embedding model, retrieval parameters, and
  prompt templates.

THE THREE METRICS WE IMPLEMENT:

  1. **Context Relevancy** — Are the retrieved chunks actually relevant to
     the question?  Measures retrieval quality.
     Score: 0-1 where 1 = all chunks are perfectly relevant.

  2. **Answer Faithfulness** — Is the answer grounded in the retrieved context?
     Detects hallucination.
     Score: 0-1 where 1 = every claim in the answer is supported by context.

  3. **Answer Relevancy** — Does the answer actually address the question?
     Measures generation quality.
     Score: 0-1 where 1 = the answer perfectly addresses the question.

HOW THESE RELATE TO THE RAGAS FRAMEWORK:
  These metrics are inspired by RAGAS (Retrieval Augmented Generation
  Assessment), the leading open-source RAG evaluation framework.  Our
  implementations are simplified educational versions — they use embedding
  similarity rather than LLM-as-judge, making them fast and free.

  For production evaluation, consider the full RAGAS library:
    pip install ragas
    Docs: https://docs.ragas.io/

EVALUATION WITHOUT AN LLM (our approach):
  Full RAGAS uses an LLM to judge answer quality (LLM-as-judge).  This is
  accurate but slow and expensive.  Our implementations use embedding-based
  similarity instead:
    - Context relevancy: cosine similarity between query and each chunk.
    - Answer faithfulness: cosine similarity between answer sentences and context.
    - Answer relevancy: cosine similarity between question and answer.

  This is much faster (no LLM calls!) and provides directionally correct
  signals for iterating on your RAG system.

LEARNING RESOURCES:
  - RAGAS docs:
    https://docs.ragas.io/
  - RAGAS paper: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
    (Es et al., 2023) — https://arxiv.org/abs/2309.15217
  - VIDEO: "Evaluating RAG Pipelines with RAGAS" (James Briggs):
    https://www.youtube.com/watch?v=25mMPVt-gRk
  - VIDEO: "RAG Evaluation — How to Measure RAG Quality" (Matt Ambrogi):
    https://www.youtube.com/watch?v=E77yMKv1fDs
  - VIDEO: "LLM Evaluation Metrics Explained" (AssemblyAI):
    https://www.youtube.com/watch?v=aCSBHOmBV0c
  - "Survey of RAG Evaluation" (Gao et al., 2024):
    https://arxiv.org/abs/2404.10981

PAPERS:
  - "RAGAS: Automated Evaluation of RAG" (Es et al., 2023):
    https://arxiv.org/abs/2309.15217
  - "Benchmarking Large Language Models in RAG" (Chen et al., 2024):
    https://arxiv.org/abs/2309.01431
  - "FaithEval: Can Your LLM Stay Faithful?" (Ming et al., 2024):
    https://arxiv.org/abs/2404.16480
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentexplorr.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Results
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Result of evaluating a single RAG response.

    Attributes:
        context_relevancy:   How relevant the retrieved chunks are to the query.
                             Range: 0.0 to 1.0 (higher = better).
        answer_faithfulness: How grounded the answer is in the retrieved context.
                             Range: 0.0 to 1.0 (higher = less hallucination).
        answer_relevancy:    How well the answer addresses the question.
                             Range: 0.0 to 1.0 (higher = more on-topic).
        details:             Detailed per-chunk and per-sentence breakdowns.
    """

    context_relevancy: float = 0.0
    answer_faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Compute an overall quality score (harmonic mean of all metrics).

        WHY HARMONIC MEAN?
          The harmonic mean penalizes low scores more than the arithmetic mean.
          If any single metric is bad (e.g. faithfulness = 0.1), the overall
          score drops sharply — which is what we want, because a hallucinating
          answer is bad regardless of how relevant the retrieval was.

          Formula: 3 / (1/a + 1/b + 1/c)
        """
        scores = [
            self.context_relevancy,
            self.answer_faithfulness,
            self.answer_relevancy,
        ]

        # Avoid division by zero — if any score is 0, overall is 0
        if any(s <= 0.0 for s in scores):
            return 0.0

        return len(scores) / sum(1.0 / s for s in scores)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"context_relevancy={self.context_relevancy:.3f}, "
            f"faithfulness={self.answer_faithfulness:.3f}, "
            f"answer_relevancy={self.answer_relevancy:.3f}, "
            f"overall={self.overall_score:.3f})"
        )


# ---------------------------------------------------------------------------
# RAG Evaluator
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """Evaluate RAG pipeline quality using embedding-based metrics.

    This evaluator computes three core metrics without requiring LLM calls,
    using embedding cosine similarity as a proxy for semantic relatedness.

    HOW IT WORKS:
      All three metrics follow the same pattern:
        1. Embed the inputs (query, context chunks, answer) using the same
           sentence-transformer model used in the RAG pipeline.
        2. Compute cosine similarity between the relevant pairs.
        3. Aggregate the similarities into a 0-1 score.

      This is fast (~100ms for a typical evaluation) and free (no API calls).

    LIMITATIONS:
      - Embedding similarity is a rough proxy for semantic relevance.
        It can miss nuances that an LLM-as-judge would catch.
      - It doesn't detect logical errors or reasoning failures.
      - For high-stakes applications, supplement with LLM-based evaluation
        (see RAGAS: https://docs.ragas.io/).

    Args:
        embedding_model: An ``EmbeddingModel`` instance.  If ``None``, creates
                         one with the default model.

    Usage::

        from agentexplorr.rag.evaluation import RAGEvaluator

        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            query="What is attention in transformers?",
            contexts=["Attention allows the model to focus on relevant parts..."],
            answer="Attention is a mechanism that allows transformers to...",
        )
        print(result)
        # EvaluationResult(context_relevancy=0.847, faithfulness=0.912,
        #                  answer_relevancy=0.891, overall=0.883)
    """

    def __init__(self, embedding_model: Any | None = None) -> None:
        if embedding_model is not None:
            self._embedding_model = embedding_model
        else:
            from agentexplorr.rag.embeddings import EmbeddingModel

            self._embedding_model = EmbeddingModel()

        logger.info("rag_evaluator_initialized")

    def evaluate(
        self,
        query: str,
        contexts: list[str],
        answer: str,
    ) -> EvaluationResult:
        """Evaluate a RAG response on all three metrics.

        Args:
            query:    The original user question.
            contexts: List of retrieved chunk texts.
            answer:   The LLM-generated answer.

        Returns:
            An EvaluationResult with all three scores and detailed breakdowns.
        """
        logger.info(
            "evaluating_rag_response",
            query_preview=query[:80],
            num_contexts=len(contexts),
            answer_length=len(answer),
        )

        # Compute each metric
        cr_score, cr_details = self.context_relevancy(query, contexts)
        af_score, af_details = self.answer_faithfulness(answer, contexts)
        ar_score, ar_details = self.answer_relevancy(query, answer)

        result = EvaluationResult(
            context_relevancy=cr_score,
            answer_faithfulness=af_score,
            answer_relevancy=ar_score,
            details={
                "context_relevancy": cr_details,
                "answer_faithfulness": af_details,
                "answer_relevancy": ar_details,
            },
        )

        logger.info(
            "evaluation_complete",
            context_relevancy=round(cr_score, 3),
            faithfulness=round(af_score, 3),
            answer_relevancy=round(ar_score, 3),
            overall=round(result.overall_score, 3),
        )

        return result

    # ------------------------------------------------------------------
    # Metric 1: Context Relevancy
    # ------------------------------------------------------------------

    def context_relevancy(
        self,
        query: str,
        contexts: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Measure how relevant the retrieved contexts are to the query.

        HOW IT WORKS:
          1. Embed the query.
          2. Embed each retrieved context chunk.
          3. Compute cosine similarity between the query and each chunk.
          4. The score is the MEAN similarity across all chunks.

        WHY MEAN?
          We want to penalize retrieving irrelevant chunks.  If you retrieve
          5 chunks but only 2 are relevant, the mean drops — signaling that
          your retrieval precision needs improvement.

        INTERPRETATION:
          - 0.8+ : Excellent — retrieved chunks are highly relevant.
          - 0.6-0.8: Good — most chunks are relevant, some noise.
          - 0.4-0.6: Fair — significant irrelevant content in retrieved chunks.
          - <0.4 : Poor — retrieval is returning mostly irrelevant chunks.

        Args:
            query:    The user's question.
            contexts: List of retrieved chunk texts.

        Returns:
            Tuple of (score, details_dict).
        """
        if not contexts:
            return 0.0, {"per_chunk_scores": [], "message": "No contexts provided"}

        # Embed query and contexts
        query_embedding = self._embedding_model.embed_text(query)
        context_embeddings = self._embedding_model.embed_batch(contexts)

        # Compute similarity between query and each context
        per_chunk_scores: list[float] = []
        for i, ctx_embedding in enumerate(context_embeddings):
            similarity = self._cosine_similarity(query_embedding, ctx_embedding)
            per_chunk_scores.append(similarity)

        # Overall score is the mean similarity
        score = sum(per_chunk_scores) / len(per_chunk_scores)

        # Clamp to [0, 1] (cosine similarity for normalized vectors is already
        # in [-1, 1], but text embeddings rarely go negative)
        score = max(0.0, min(1.0, score))

        details = {
            "per_chunk_scores": [round(s, 4) for s in per_chunk_scores],
            "num_contexts": len(contexts),
            "min_score": round(min(per_chunk_scores), 4),
            "max_score": round(max(per_chunk_scores), 4),
        }

        return score, details

    # ------------------------------------------------------------------
    # Metric 2: Answer Faithfulness
    # ------------------------------------------------------------------

    def answer_faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Measure how well the answer is grounded in the retrieved context.

        HOW IT WORKS:
          This metric detects hallucination — claims in the answer that are
          NOT supported by the retrieved context.

          1. Split the answer into sentences (each sentence = one "claim").
          2. For each sentence, find the most similar context chunk.
          3. If the similarity is high, the claim is "supported".
          4. The score is the fraction of supported claims.

        WHAT COUNTS AS "SUPPORTED"?
          A sentence is considered supported if its maximum cosine similarity
          to any context chunk exceeds a threshold (default 0.5).  This
          threshold can be tuned:
            - Lower threshold → more lenient (fewer flagged hallucinations).
            - Higher threshold → stricter (more flagged hallucinations).

        INTERPRETATION:
          - 0.9+ : Excellent — answer is well-grounded in the context.
          - 0.7-0.9: Good — mostly grounded, some unsupported statements.
          - 0.5-0.7: Concerning — significant hallucination detected.
          - <0.5 : Poor — the answer is mostly NOT grounded in the context.

        Args:
            answer:   The LLM-generated answer.
            contexts: List of retrieved chunk texts.

        Returns:
            Tuple of (score, details_dict).
        """
        if not answer.strip():
            return 0.0, {"message": "Empty answer"}
        if not contexts:
            return 0.0, {"message": "No contexts to check against"}

        # Split answer into sentences
        sentences = self._split_sentences(answer)
        if not sentences:
            return 0.0, {"message": "Could not split answer into sentences"}

        # Combine all contexts into one text for embedding (we also embed
        # each context separately for more granular matching)
        context_embeddings = self._embedding_model.embed_batch(contexts)
        sentence_embeddings = self._embedding_model.embed_batch(sentences)

        # For each sentence, find the maximum similarity to any context chunk
        support_threshold = 0.5
        sentence_scores: list[dict[str, Any]] = []
        supported_count = 0

        for sent_idx, sent_embedding in enumerate(sentence_embeddings):
            # Compare this sentence against all context chunks
            max_similarity = 0.0
            best_context_idx = 0

            for ctx_idx, ctx_embedding in enumerate(context_embeddings):
                similarity = self._cosine_similarity(sent_embedding, ctx_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_context_idx = ctx_idx

            is_supported = max_similarity >= support_threshold
            if is_supported:
                supported_count += 1

            sentence_scores.append({
                "sentence": sentences[sent_idx][:100],
                "max_similarity": round(max_similarity, 4),
                "best_context_idx": best_context_idx,
                "is_supported": is_supported,
            })

        # Score = fraction of supported sentences
        score = supported_count / len(sentences)

        details = {
            "total_sentences": len(sentences),
            "supported_sentences": supported_count,
            "unsupported_sentences": len(sentences) - supported_count,
            "support_threshold": support_threshold,
            "per_sentence": sentence_scores,
        }

        return score, details

    # ------------------------------------------------------------------
    # Metric 3: Answer Relevancy
    # ------------------------------------------------------------------

    def answer_relevancy(
        self,
        query: str,
        answer: str,
    ) -> tuple[float, dict[str, Any]]:
        """Measure how well the answer addresses the original question.

        HOW IT WORKS:
          This is the simplest metric: we compute the cosine similarity
          between the embeddings of the question and the answer.

          A relevant answer should be semantically close to the question
          (it talks about the same topic and directly addresses it).

          An irrelevant answer might be factually correct but off-topic
          (e.g. Q: "What is attention?" A: "Python is a programming language.")

        REFINEMENT — SENTENCE-LEVEL SCORING:
          For longer answers, we also compute per-sentence similarity to the
          query.  This helps identify which parts of the answer are relevant
          and which are padding/tangents.

        INTERPRETATION:
          - 0.8+ : Excellent — answer directly addresses the question.
          - 0.6-0.8: Good — answer is mostly on-topic.
          - 0.4-0.6: Fair — answer partially addresses the question.
          - <0.4 : Poor — answer is off-topic.

        Args:
            query:  The original user question.
            answer: The LLM-generated answer.

        Returns:
            Tuple of (score, details_dict).
        """
        if not answer.strip():
            return 0.0, {"message": "Empty answer"}
        if not query.strip():
            return 0.0, {"message": "Empty query"}

        # Compute overall similarity between question and answer
        query_embedding = self._embedding_model.embed_text(query)
        answer_embedding = self._embedding_model.embed_text(answer)

        overall_similarity = self._cosine_similarity(query_embedding, answer_embedding)

        # Also compute per-sentence similarity for granular analysis
        sentences = self._split_sentences(answer)
        per_sentence_scores: list[dict[str, Any]] = []

        if sentences:
            sentence_embeddings = self._embedding_model.embed_batch(sentences)
            for sent_idx, sent_embedding in enumerate(sentence_embeddings):
                sim = self._cosine_similarity(query_embedding, sent_embedding)
                per_sentence_scores.append({
                    "sentence": sentences[sent_idx][:100],
                    "relevancy_score": round(sim, 4),
                })

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, overall_similarity))

        details = {
            "overall_similarity": round(overall_similarity, 4),
            "per_sentence": per_sentence_scores,
        }

        return score, details

    # ------------------------------------------------------------------
    # Batch Evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        queries: list[str],
        contexts_list: list[list[str]],
        answers: list[str],
    ) -> list[EvaluationResult]:
        """Evaluate a batch of RAG responses.

        Useful for running evaluations across a test set.

        Args:
            queries:       List of questions.
            contexts_list: List of context lists (one per query).
            answers:       List of answers (one per query).

        Returns:
            List of EvaluationResult objects.

        Raises:
            ValueError: If the input lists have different lengths.
        """
        if not (len(queries) == len(contexts_list) == len(answers)):
            raise ValueError(
                f"Input lists must have the same length. Got: "
                f"queries={len(queries)}, contexts={len(contexts_list)}, "
                f"answers={len(answers)}"
            )

        results: list[EvaluationResult] = []
        for i, (query, contexts, answer) in enumerate(
            zip(queries, contexts_list, answers)
        ):
            logger.debug("evaluating_batch_item", index=i, total=len(queries))
            result = self.evaluate(query, contexts, answer)
            results.append(result)

        # Log aggregate statistics
        if results:
            avg_cr = sum(r.context_relevancy for r in results) / len(results)
            avg_af = sum(r.answer_faithfulness for r in results) / len(results)
            avg_ar = sum(r.answer_relevancy for r in results) / len(results)
            avg_overall = sum(r.overall_score for r in results) / len(results)

            logger.info(
                "batch_evaluation_complete",
                count=len(results),
                avg_context_relevancy=round(avg_cr, 3),
                avg_faithfulness=round(avg_af, 3),
                avg_answer_relevancy=round(avg_ar, 3),
                avg_overall=round(avg_overall, 3),
            )

        return results

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Formula: cos(theta) = (A . B) / (||A|| * ||B||)

        We implement this with basic Python to avoid hard dependencies.
        For large-scale evaluation, you would use numpy for vectorized ops.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a * a for a in vec_a) ** 0.5
        magnitude_b = sum(b * b for b in vec_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex heuristics.

        Handles common sentence boundaries:
          - Period + space + capital letter
          - Question marks and exclamation marks
          - Newlines

        For production use, consider nltk.sent_tokenize() or spaCy.
        """
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        # Filter out empty strings and very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def __repr__(self) -> str:
        return f"RAGEvaluator(model={self._embedding_model.model_name!r})"

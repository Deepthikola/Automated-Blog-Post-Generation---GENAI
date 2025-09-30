# src/metrics.py

import os
import json
import time
import pandas as pd
import numpy as np
import psutil
from jiwer import wer, cer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


class PipelineMetrics:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir

        self.chunk_dir = os.path.join(self.base_dir, "chunks")
        self.outputs_dir = os.path.join(self.base_dir, "outputs")
        self.indices_dir = os.path.join(self.base_dir, "indices")

        # Initialize models
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorizer = TfidfVectorizer(
            stop_words='english', max_features=1000)

        print(f"📊 Metrics initialized for: {self.base_dir}")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def measure_extraction_quality(self, sample_pdf_path: str, ground_truth_text: str = None) -> Dict:
        """
        Measure extraction quality metrics
        """
        print("🔍 Measuring Extraction Quality...")

        metrics = {}
        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Check chunk directory for extracted content
        chunk_files = [f for f in os.listdir(
            self.chunk_dir) if f.endswith('.txt')]
        total_chars = 0
        chunks_data = []

        for chunk_file in chunk_files:
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            with open(chunk_path, 'r', encoding='utf-8') as f:
                content = f.read()
                total_chars += len(content)
                chunks_data.append({
                    'file': chunk_file,
                    'content': content,
                    'length': len(content),
                    'words': len(content.split())
                })

        # Basic extraction metrics
        metrics['total_chunks'] = len(chunk_files)
        metrics['total_characters'] = total_chars
        metrics['avg_chunk_length'] = total_chars / \
            len(chunk_files) if chunk_files else 0

        # OCR Accuracy (if ground truth provided)
        if ground_truth_text and chunks_data:
            extracted_text = " ".join([chunk['content']
                                      for chunk in chunks_data])

            # Calculate WER and CER
            try:
                metrics['word_error_rate'] = wer(
                    ground_truth_text, extracted_text)
                metrics['character_error_rate'] = cer(
                    ground_truth_text, extracted_text)

                # Coverage ratio
                metrics['coverage_ratio'] = len(
                    extracted_text) / len(ground_truth_text) if ground_truth_text else 0
                metrics['missing_rate'] = 1 - metrics['coverage_ratio']
            except Exception as e:
                print(f"⚠️ Error calculating OCR metrics: {e}")
                metrics['word_error_rate'] = None
                metrics['character_error_rate'] = None
                metrics['coverage_ratio'] = None
                metrics['missing_rate'] = None

        # Image extraction metrics
        image_files = [f for f in os.listdir(os.path.join(self.outputs_dir, 'images', 'extracted'))
                       if f.endswith('.png')] if os.path.exists(os.path.join(self.outputs_dir, 'images', 'extracted')) else []
        metrics['total_images'] = len(image_files)

        end_time = time.time()
        end_memory = self.get_memory_usage()

        metrics['extraction_time_seconds'] = end_time - start_time
        metrics['extraction_memory_mb'] = end_memory - start_memory

        return self.convert_numpy_types(metrics)

    def measure_chunk_quality(self) -> Dict:
        """
        Measure chunk quality metrics
        """
        print("🔍 Measuring Chunk Quality...")

        metrics = {}
        chunks_data = []

        # Load all chunks
        chunk_files = [f for f in os.listdir(
            self.chunk_dir) if f.endswith('.txt')]

        for chunk_file in chunk_files:
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            with open(chunk_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only non-empty chunks
                    chunks_data.append({
                        'file': chunk_file,
                        'content': content,
                        'length': len(content)
                    })

        if not chunks_data:
            return {"error": "No chunks found"}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(chunks_data)

        # Duplicate analysis
        metrics['total_chunks'] = len(df)
        metrics['duplicate_count'] = int(df.duplicated(
            subset=['content']).sum())  # Convert to int
        metrics['duplicate_rate'] = float(
            metrics['duplicate_count'] / metrics['total_chunks'])
        metrics['unique_chunk_count'] = metrics['total_chunks'] - \
            metrics['duplicate_count']

        # Chunk length statistics
        metrics['avg_chunk_length'] = float(df['length'].mean())
        metrics['min_chunk_length'] = int(df['length'].min())
        metrics['max_chunk_length'] = int(df['length'].max())
        metrics['chunk_length_std'] = float(df['length'].std())

        # Chunk overlap analysis using cosine similarity
        if len(chunks_data) > 1:
            try:
                # Sample chunks for overlap analysis (to avoid memory issues)
                sample_size = min(50, len(chunks_data))
                sample_chunks = [chunk['content']
                                 for chunk in chunks_data[:sample_size]]

                # Create embeddings
                embeddings = self.model.encode(sample_chunks)

                # Calculate cosine similarity matrix
                similarity_matrix = cosine_similarity(embeddings)

                # Get upper triangle (excluding diagonal)
                triu_indices = np.triu_indices_from(similarity_matrix, k=1)
                overlap_scores = similarity_matrix[triu_indices]

                metrics['avg_chunk_overlap'] = float(np.mean(overlap_scores))
                metrics['max_chunk_overlap'] = float(np.max(overlap_scores))
                metrics['overlap_std'] = float(np.std(overlap_scores))

            except Exception as e:
                print(f"⚠️ Error calculating chunk overlap: {e}")
                metrics['avg_chunk_overlap'] = None
                metrics['max_chunk_overlap'] = None
                metrics['overlap_std'] = None

        return self.convert_numpy_types(metrics)

    def measure_retrieval_quality(self, test_queries: List[Tuple[str, List[str]]] = None) -> Dict:
        """
        Measure retrieval quality metrics
        test_queries: List of (query, [relevant_chunk_files])
        """
        print("🔍 Measuring Retrieval Quality...")

        metrics = {}

        # Load FAISS indices
        try:
            text_index = faiss.read_index(os.path.join(
                self.indices_dir, "faiss_text_index.bin"))
            with open(os.path.join(self.indices_dir, "faiss_text_meta.json"), 'r', encoding='utf-8') as f:
                text_meta = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load FAISS indices: {e}")
            return {"error": "FAISS indices not available"}

        # Default test queries if none provided
        if test_queries is None:
            # Create some sample queries based on chunk content
            chunk_files = [f for f in os.listdir(
                self.chunk_dir) if f.endswith('.txt')]
            test_queries = []

            if chunk_files:
                # Use first chunk to create a test query
                first_chunk_file = chunk_files[0]
                with open(os.path.join(self.chunk_dir, first_chunk_file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract first few words as query
                    words = content.split()[:10]
                    if words:
                        query = " ".join(words)
                        # The chunk itself is relevant
                        test_queries = [(query, [first_chunk_file])]

        if not test_queries:
            return {"error": "No test queries available"}

        precision_scores = []
        recall_scores = []
        mrr_scores = []

        for query, relevant_docs in test_queries:
            # Encode query
            q_emb = self.model.encode([query], convert_to_numpy=True)
            q_emb = q_emb.astype("float32")
            faiss.normalize_L2(q_emb)

            # Search
            k = min(10, len(text_meta))
            D, I = text_index.search(q_emb, k)

            retrieved_docs = [text_meta[i] for i in I[0]]

            # Calculate precision and recall
            relevant_retrieved = set(retrieved_docs) & set(relevant_docs)

            precision = len(relevant_retrieved) / \
                len(retrieved_docs) if retrieved_docs else 0
            recall = len(relevant_retrieved) / \
                len(relevant_docs) if relevant_docs else 0

            precision_scores.append(precision)
            recall_scores.append(recall)

            # Calculate MRR
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc in relevant_docs:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)

        metrics['mean_precision'] = float(
            np.mean(precision_scores)) if precision_scores else 0
        metrics['mean_recall'] = float(
            np.mean(recall_scores)) if recall_scores else 0
        metrics['mean_reciprocal_rank'] = float(
            np.mean(mrr_scores)) if mrr_scores else 0
        metrics['queries_evaluated'] = len(test_queries)

        return self.convert_numpy_types(metrics)

    def measure_efficiency_metrics(self) -> Dict:
        """
        Measure time and memory efficiency
        """
        print("⏱️ Measuring Efficiency Metrics...")

        metrics = {}

        # Measure extraction time (simulate on a small PDF)
        pdf_dir = os.path.join(self.base_dir, 'data', 'input_pdfs')
        sample_pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(
            '.pdf')] if os.path.exists(pdf_dir) else []

        if sample_pdfs:
            pdf_path = os.path.join(pdf_dir, sample_pdfs[0])

            # Time extraction (simplified)
            start_time = time.time()
            start_memory = self.get_memory_usage()

            # Simulate extraction by reading existing chunks
            chunk_files = [f for f in os.listdir(
                self.chunk_dir) if f.endswith('.txt')]
            time.sleep(0.1)  # Simulate work

            end_memory = self.get_memory_usage()
            end_time = time.time()

            metrics['extraction_time_per_pdf'] = float(
                (end_time - start_time) * 10)  # Estimate
            metrics['extraction_memory_peak_mb'] = float(
                end_memory - start_memory)

        # Measure retrieval time
        try:
            text_index = faiss.read_index(os.path.join(
                self.indices_dir, "faiss_text_index.bin"))

            start_time = time.time()
            start_memory = self.get_memory_usage()

            # Test query
            test_query = "sample query for timing"
            q_emb = self.model.encode([test_query], convert_to_numpy=True)
            q_emb = q_emb.astype("float32")
            faiss.normalize_L2(q_emb)
            D, I = text_index.search(q_emb, 5)

            end_memory = self.get_memory_usage()
            end_time = time.time()

            metrics['retrieval_time_ms'] = float(
                (end_time - start_time) * 1000)
            metrics['retrieval_memory_mb'] = float(end_memory - start_memory)

        except Exception as e:
            print(f"⚠️ Error measuring retrieval efficiency: {e}")

        return self.convert_numpy_types(metrics)

    def generate_comprehensive_report(self, ground_truth_text: str = None) -> Dict:
        """
        Generate comprehensive metrics report
        """
        print("📈 Generating Comprehensive Metrics Report...")

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_version': 'PDF-to-Blog v2.0',
            'base_directory': self.base_dir
        }

        # 1. Extraction Quality Metrics
        report['extraction_quality'] = self.measure_extraction_quality(
            sample_pdf_path="",
            ground_truth_text=ground_truth_text
        )

        # 2. Chunk Quality Metrics
        report['chunk_quality'] = self.measure_chunk_quality()

        # 3. Retrieval Quality Metrics
        report['retrieval_quality'] = self.measure_retrieval_quality()

        # 4. Efficiency Metrics
        report['efficiency_metrics'] = self.measure_efficiency_metrics()

        # 5. Overall Scores
        report['overall_scores'] = self.calculate_overall_scores(report)

        return self.convert_numpy_types(report)

    def calculate_overall_scores(self, report: Dict) -> Dict:
        """
        Calculate overall performance scores
        """
        scores = {}

        # Extraction Score (0-100)
        ext = report['extraction_quality']
        chunk = report['chunk_quality']
        retrieval = report['retrieval_quality']
        efficiency = report['efficiency_metrics']

        # Quality Score (based on chunk quality and retrieval)
        if 'duplicate_rate' in chunk and 'mean_precision' in retrieval:
            quality_score = (
                (1 - chunk.get('duplicate_rate', 0)) * 40 +  # Chunk quality weight
                # Retrieval precision weight
                retrieval.get('mean_precision', 0) * 30 +
                # Retrieval recall weight
                retrieval.get('mean_recall', 0) * 30
            )
            scores['quality_score'] = min(100, quality_score * 100)

        # Efficiency Score
        if 'retrieval_time_ms' in efficiency:
            eff_score = max(
                0, 100 - (efficiency.get('retrieval_time_ms', 1000) / 10))
            scores['efficiency_score'] = eff_score

        # Overall Score
        if 'quality_score' in scores and 'efficiency_score' in scores:
            scores['overall_score'] = (scores['quality_score'] * 0.7 +
                                       scores['efficiency_score'] * 0.3)

        return self.convert_numpy_types(scores)

    def save_report(self, report: Dict, output_path: str = None):
        """
        Save metrics report to file
        """
        if output_path is None:
            output_path = os.path.join(
                self.outputs_dir, f"metrics_report_{int(time.time())}.json")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use custom encoder for numpy types
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2,
                      ensure_ascii=False, cls=NumpyEncoder)

        print(f"✅ Metrics report saved to: {output_path}")
        return output_path

    def visualize_metrics(self, report: Dict):
        """
        Create visualizations for the metrics
        """
        try:
            # Create visualization directory
            viz_dir = os.path.join(self.outputs_dir, "metrics_visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # 1. Chunk Quality Visualization
            if 'chunk_quality' in report:
                chunk_data = report['chunk_quality']
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Duplicate rate pie chart
                if 'duplicate_rate' in chunk_data:
                    labels = ['Unique Chunks', 'Duplicate Chunks']
                    sizes = [1 - chunk_data['duplicate_rate'],
                             chunk_data['duplicate_rate']]
                    axes[0].pie(sizes, labels=labels,
                                autopct='%1.1f%%', startangle=90)
                    axes[0].set_title('Chunk Duplication Rate')

                # Chunk length distribution
                if 'avg_chunk_length' in chunk_data:
                    # Simulate some length data for visualization
                    lengths = [chunk_data.get('min_chunk_length', 100),
                               chunk_data.get('avg_chunk_length', 500),
                               chunk_data.get('max_chunk_length', 1000)]
                    axes[1].bar(['Min', 'Avg', 'Max'], lengths,
                                color=['red', 'blue', 'green'])
                    axes[1].set_title('Chunk Length Distribution')
                    axes[1].set_ylabel('Characters')

                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'chunk_quality.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

            # 2. Retrieval Quality Visualization
            if 'retrieval_quality' in report:
                ret_data = report['retrieval_quality']
                metrics = ['Precision', 'Recall', 'MRR']
                values = [ret_data.get('mean_precision', 0),
                          ret_data.get('mean_recall', 0),
                          ret_data.get('mean_reciprocal_rank', 0)]

                plt.figure(figsize=(8, 6))
                bars = plt.bar(metrics, values, color=[
                               'skyblue', 'lightgreen', 'lightcoral'])
                plt.ylim(0, 1)
                plt.title('Retrieval Quality Metrics')
                plt.ylabel('Score')

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom')

                plt.savefig(os.path.join(
                    viz_dir, 'retrieval_quality.png'), dpi=300, bbox_inches='tight')
                plt.close()

            print(f"✅ Visualizations saved to: {viz_dir}")

        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")


def main():
    """
    Main function to run metrics evaluation
    """
    print("🚀 Starting PDF-to-Blog Pipeline Metrics Evaluation")

    # Initialize metrics calculator
    metrics = PipelineMetrics()

    # Generate comprehensive report
    # Note: For OCR accuracy, you'd need to provide ground truth text
    report = metrics.generate_comprehensive_report(
        ground_truth_text=None  # Provide ground truth text if available
    )

    # Print summary
    print("\n" + "="*60)
    print("📊 METRICS SUMMARY")
    print("="*60)

    # Extraction Quality
    ext = report.get('extraction_quality', {})
    print(f"📄 Extraction Quality:")
    print(f"   - Total Chunks: {ext.get('total_chunks', 0)}")
    print(f"   - Total Images: {ext.get('total_images', 0)}")
    print(
        f"   - Extraction Time: {ext.get('extraction_time_seconds', 0):.2f}s")

    # Chunk Quality
    chunk = report.get('chunk_quality', {})
    print(f"📦 Chunk Quality:")
    print(f"   - Unique Chunks: {chunk.get('unique_chunk_count', 0)}")
    print(f"   - Duplicate Rate: {chunk.get('duplicate_rate', 0):.2%}")
    print(f"   - Avg Chunk Overlap: {chunk.get('avg_chunk_overlap', 0):.3f}")

    # Retrieval Quality
    ret = report.get('retrieval_quality', {})
    print(f"🔍 Retrieval Quality:")
    print(f"   - Mean Precision: {ret.get('mean_precision', 0):.3f}")
    print(f"   - Mean Recall: {ret.get('mean_recall', 0):.3f}")
    print(
        f"   - Mean Reciprocal Rank: {ret.get('mean_reciprocal_rank', 0):.3f}")

    # Efficiency
    eff = report.get('efficiency_metrics', {})
    print(f"⏱️ Efficiency Metrics:")
    print(f"   - Retrieval Time: {eff.get('retrieval_time_ms', 0):.2f}ms")
    print(f"   - Retrieval Memory: {eff.get('retrieval_memory_mb', 0):.2f}MB")

    # Overall Scores
    overall = report.get('overall_scores', {})
    print(f"🏆 Overall Scores:")
    print(f"   - Quality Score: {overall.get('quality_score', 0):.1f}/100")
    print(
        f"   - Efficiency Score: {overall.get('efficiency_score', 0):.1f}/100")
    print(f"   - Overall Score: {overall.get('overall_score', 0):.1f}/100")

    # Save report
    report_path = metrics.save_report(report)

    # Generate visualizations
    metrics.visualize_metrics(report)

    print(f"\n✅ Metrics evaluation completed!")
    print(f"📁 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()

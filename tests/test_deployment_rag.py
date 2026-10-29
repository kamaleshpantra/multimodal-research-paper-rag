import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'retrieval'))
import deployment_rag as rag
import build_deployment_index as deployment_index_builder


class FakeEmbedder:
    def embed(self, texts):
        for _ in texts:
            yield [1.0, 0.0, 0.0]


class DeploymentRagTests(unittest.TestCase):
    def test_unreviewed_visuals_are_excluded(self):
        self.assertTrue(rag.is_rejected_visual({'source_type': 'visual_description', 'metadata': {'review_status': 'pending_manual_review', 'approved_for_index': False}}))
        self.assertTrue(rag.is_rejected_visual({'source_type': 'visual_description', 'metadata': {'review_status': 'rejected_after_quality_audit', 'approved_for_index': False}}))
        self.assertFalse(rag.is_rejected_visual({'source_type': 'visual_description', 'metadata': {'review_status': 'approved_after_manual_quality_audit', 'approved_for_index': True}}))
        self.assertTrue(rag.is_rejected_visual({'source_type': 'visual_description', 'metadata': {'review_status': 'approved_by_name_only', 'approved_for_index': False}}))
        self.assertFalse(rag.is_rejected_visual({'source_type': 'figure_caption', 'metadata': {}}))

    def test_exact_bibliography_title_wins_over_similar_title(self):
        records = [
            {'paper_id': 'p', 'source_type': 'bibliography_reference', 'chunk_id': 'other', 'metadata': {'title': 'Personalized transformer for explainable recommendation'}},
            {'paper_id': 'p', 'source_type': 'bibliography_reference', 'chunk_id': 'exact', 'metadata': {'title': 'Personalized prompt learning for explainable recommendation'}},
        ]
        question = 'Which earlier work in the bibliography is titled Personalized prompt learning for explainable recommendation?'
        got = rag.find_exact_bibliography_match(question, records)
        self.assertEqual(got['mode'], 'exact_match')
        self.assertEqual(got['record']['chunk_id'], 'exact')
        missing = rag.find_exact_bibliography_match('Which bibliography is titled A Completely Different Long Work?', records)
        self.assertEqual(missing['mode'], 'no_exact_match')

    def test_deduplication_removes_cross_pipeline_copy_but_not_figure(self):
        text = ('The retriever conditions on the input query and searches the document index using maximum inner product search. '
                'The selected passages are then supplied to the generator with the original query and prior output tokens. '
                'This shared passage is intentionally long enough for robust duplicate detection in the test fixture.')
        records = [
            {'paper_id': 'p', 'source_type': 'page_text', 'chunk_id': 'page-copy', 'text': text + ' footer text'},
            {'paper_id': 'p', 'source_type': 'section_text', 'chunk_id': 'section-clean', 'text': text},
            {'paper_id': 'p', 'source_type': 'visual_description', 'chunk_id': 'figure', 'text': text, 'metadata': {'approved_for_index': True, 'review_status': 'approved'}},
        ]
        kept, removed = rag.deduplicate_retrieved(records)
        ids = [r['chunk_id'] for r in kept]
        self.assertIn('section-clean', ids)
        self.assertIn('figure', ids)
        self.assertNotIn('page-copy', ids)
        self.assertEqual(len(removed), 1)

    def test_builder_requires_explicit_visual_approval(self):
        self.assertTrue(deployment_index_builder.approved_record({'source_type': 'figure_caption', 'metadata': {}}))
        self.assertTrue(deployment_index_builder.approved_record({'source_type': 'visual_description', 'metadata': {'approved_for_index': True}}))
        self.assertFalse(deployment_index_builder.approved_record({'source_type': 'visual_description', 'metadata': {'review_status': 'approved_by_name_only', 'approved_for_index': False}}))
        self.assertFalse(deployment_index_builder.approved_record({'source_type': 'visual_description', 'metadata': {'review_status': 'pending_manual_review'}}))

    def test_explicit_parameter_scan_finds_assignment_not_year(self):
        question = 'What exact learning rate value was used?'
        records = [
            {'paper_id': 'p', 'source_type': 'section_text', 'chunk_id': 'year', 'section': 'Training', 'text': 'In 2024 the authors discuss a learning rate in general.'},
            {'paper_id': 'p', 'source_type': 'section_text', 'chunk_id': 'value', 'section': 'Training', 'text': 'The learning rate was set to 3e-5 for training.'},
        ]
        hits, scan = rag.scan_numeric_assignments(question, records)
        self.assertTrue(scan['triggered'])
        self.assertEqual(hits[0]['chunk_id'], 'value')
        self.assertEqual(hits[0]['numeric_parameter_scan']['matched_value'], '3e-5')

    def test_library_wide_parameter_scan_keeps_scientific_subject_terms(self):
        question = 'What exact SAE sparsity coefficient was used in experiments described by papers in this library?'
        records = [{'paper_id': 'p', 'source_type': 'section_text', 'chunk_id': 'sae-value', 'section': 'SAE experiments',
                    'text': 'The SAE sparsity coefficient was set to 0.005 in the experiments.'}]
        hits, report = rag.scan_numeric_assignments(question, records)
        self.assertEqual(report['candidate_count'], 1, report)
        self.assertEqual(hits[0]['numeric_parameter_scan']['matched_value'], '0.005')

    def test_citation_markers_are_validated(self):
        cleaned, warnings = rag.validate_citations('The result is explained here [S1, S9].', 2)
        self.assertIn('[S1]', cleaned)
        self.assertNotIn('S9', cleaned)
        self.assertTrue(warnings)
        cleaned, warnings = rag.validate_citations('A factual claim without a source marker.', 2)
        self.assertTrue(any('lack inline source markers' in w for w in warnings))

    def test_long_exact_overlap_flag_is_not_plagiarism_verdict(self):
        source = 'The method selects relevant documents conditioned on the input query and supplies those passages to a generator for producing the output sequence.'
        answer = 'The method selects relevant documents conditioned on the input query and supplies those passages to a generator for producing the output sequence [S1].'
        evidence = [{'paper_id': 'p', 'source_type': 'section_text', 'chunk_id': 'source', 'text': source}]
        flags = rag._exact_sentence_overlap_flags(answer, evidence, min_words=12)
        self.assertTrue(flags)
        self.assertIn('not an AI-text detector', flags[0]['warning'])

    def test_cross_paper_synthesis_balances_scope_dynamically(self):
        class FakeRetriever:
            records = [{'paper_id': p} for p in ('paper_c', 'paper_a', 'paper_b')]
            registry = {}
            def search(self, query, top_k=8, paper_id=None):
                return [
                    {'paper_id': paper_id, 'chunk_id': f'{paper_id}-{rank}', 'source_type': 'section_text',
                     'text': f'{paper_id} evidence {rank}', 'retrieval': {'adjusted_score': 1.0 / rank}}
                    for rank in range(1, 4)
                ]
        pool, info = rag.search_with_paper_coverage('Compare the central contributions across the papers', FakeRetriever(), 8, None)
        self.assertTrue(info['enabled'], info)
        self.assertEqual({pool[i]['paper_id'] for i in range(3)}, {'paper_a', 'paper_b', 'paper_c'})
        self.assertEqual(info['paper_ids'], ['paper_a', 'paper_b', 'paper_c'])

    def test_named_paper_comparison_only_balances_named_papers(self):
        class FakeRetriever:
            records = [{'paper_id': p} for p in ('paper_c', 'paper_a', 'paper_b')]
            registry = {
                'paper_a': {'title': 'RAG paper retrieval generation'},
                'paper_b': {'title': 'XRec recommendations'},
                'paper_c': {'title': 'Mechanistic interpretability review'},
            }
            def search(self, query, top_k=8, paper_id=None):
                return [{'paper_id': paper_id, 'chunk_id': f'{paper_id}-1', 'source_type': 'section_text', 'text': 'evidence', 'retrieval': {'adjusted_score': 1.0}}]
        pool, info = rag.search_with_paper_coverage('Compare RAG and XRec', FakeRetriever(), 8, None)
        self.assertTrue(info['enabled'], info)
        self.assertEqual(set(info['paper_ids']), {'paper_a', 'paper_b'})
        self.assertEqual({r['paper_id'] for r in pool}, {'paper_a', 'paper_b'})

    def test_all_paper_scope_works_for_cross_paper_summary_without_compare_word(self):
        class FakeRetriever:
            records = [{'paper_id': p} for p in ('paper_c', 'paper_a', 'paper_b')]
            registry = {}
            def search(self, query, top_k=8, paper_id=None):
                return [
                    {'paper_id': paper_id, 'chunk_id': f'{paper_id}-1', 'source_type': 'section_text',
                     'text': f'{paper_id} evidence', 'retrieval': {'adjusted_score': 1.0}}
                ]
        pool, info = rag.search_with_paper_coverage('Summarize limitations reported across these three papers', FakeRetriever(), 8, None)
        self.assertTrue(info['enabled'], info)
        self.assertEqual(set(info['paper_ids']), {'paper_a', 'paper_b', 'paper_c'})
        self.assertEqual({r['paper_id'] for r in pool}, {'paper_a', 'paper_b', 'paper_c'})

    def test_index_shape_validation(self):
        with self.assertRaises(ValueError):
            rag.DeploymentRetriever([{'text': 'x'}], __import__('numpy').zeros((2, 3), dtype='float32'), FakeEmbedder())


if __name__ == '__main__':
    unittest.main(verbosity=2)

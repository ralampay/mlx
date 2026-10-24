"""Lightweight CLI option declarations; shared flags retain historical defaults."""
import argparse
from mlx.core.cli_options import CliOption

OPTIONS = (
    CliOption(('--embedding-backend',), {
        'default': 'llama-cpp',
        'dest': 'embedding_backend',
        'help': 'Text-embedding backend; use ls-embedding-backends for discovery.',
    }),
    CliOption(('--metrics',), {
        'default': None,
        'help': 'Comma-separated retrieval metric names; use text-embedding ls-metrics for discovery.',
    }),
    CliOption(('--vector-store',), {
        'default': None,
        'dest': 'vector_store',
        'help': 'Persistent vector-store provider for text embedding.',
    }),
    CliOption(('--top-k',), {
        'type': int,
        'default': 100,
        'dest': 'top_k',
        'help': 'Maximum retrieval depth for text-embedding benchmarks.',
    }),
    CliOption(('--k-values',), {
        'default': '1,5,10,20,100',
        'dest': 'k_values',
        'help': 'Comma-separated retrieval metric cutoffs.',
    }),
    CliOption(('--pooling',), {
        'choices': ('auto', 'mean', 'cls', 'last', 'none'),
        'default': 'auto',
        'help': 'GGUF pooling: auto, mean, cls, last, none.',
    }),
    CliOption(('--prompt-format',), {
        'choices': ('auto', 'none', 'e5', 'embeddinggemma', 'qwen3'),
        'default': 'auto',
        'help': 'Retrieval text format: auto (none), none, e5, embeddinggemma, qwen3.',
    }),
    CliOption(('--context-length',), {
        'type': int,
        'default': None,
        'help': 'Optional GGUF embedding context length; the autoencoder experiment defaults to 2048.',
    }),
    CliOption(('--suite',), {
        'default': 'laptop-ae-v1',
        'help': 'Pinned retrieval suite; opt in to downloading missing NanoBEIR datasets.',
    }),
    CliOption(('--download-datasets',), {
        'action': 'store_true',
    }),
    CliOption(('--resume',), {
        'action': 'store_true',
        'help': 'Reuse verified completed retrieval experiment stages.',
    }),
    CliOption(('--exclude-self-matches',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'help': 'Exclude equal query/document IDs; enabled by default in the autoencoder experiment.',
    }),
    CliOption(('--experiment-config',), {
        'default': None,
        'help': 'Named v2 experiment JSON and verified read-only original embedding source.',
    }),
    CliOption(('--embedding-source',), {
        'default': None,
    }),
    CliOption(('--dry-run',), {
        'action': 'store_true',
        'help': 'Validate an autoencoder experiment and report counts without creating output.',
    }),
    CliOption(('--autoencoder-model',), {
        'default': 'simple',
        'help': 'Architecture and loss ablation for benchmark-autoencoders.',
    }),
    CliOption(('--losses',), {
        'default': 'mse,mse-similarity',
    }),
    CliOption(('--similarity-weight',), {
        'type': float,
        'default': 1.0,
        'help': 'Geometry weight for mse-similarity in the retrieval experiment.',
    }),
    CliOption(('--bottleneck-dims',), {
        'default': '128,256',
        'help': 'Comma-separated experiment dimensions and matched training seeds.',
    }),
    CliOption(('--seeds',), {
        'default': '42,43,44,45,46',
    }),
    CliOption(('--embedding-batch-size',), {
        'type': int,
        'default': 1,
        'help': 'Text embedding batch size, separate from autoencoder training batches.',
    }),
    CliOption(('--primary-metric',), {
        'default': 'ndcg@10',
        'help': 'Dataset-level non-inferiority outcome, absolute margin, and familywise significance level.',
    }),
    CliOption(('--noninferiority-margin',), {
        'type': float,
        'default': 0.01,
    }),
    CliOption(('--alpha',), {
        'type': float,
        'default': 0.05,
    }),
    CliOption(('--query-prefix',), {
        'default': '',
        'dest': 'query_prefix',
        'help': 'Text prepended to every retrieval query before embedding.',
    }),
    CliOption(('--document-prefix',), {
        'default': '',
        'dest': 'document_prefix',
        'help': 'Text prepended to every corpus document before embedding.',
    }),
    CliOption(('--normalize-embeddings',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'dest': 'normalize_embeddings',
        'help': 'L2-normalize vectors before export and indexing.',
    }),
    CliOption(('--representation',), {
        'default': None,
        'help': 'Generic embedding representation label recorded in research artifacts.',
    }),
    CliOption(('--model-file',), {
        'default': None,
        'dest': 'model_file',
        'help': 'GGUF embedding model used by NLP embed.',
    }),
    CliOption(('--input-file',), {
        'default': None,
        'dest': 'input_file',
        'help': 'Input CSV used by NLP embed.',
    }),
    CliOption(('--output-file',), {
        'default': None,
        'dest': 'output_file',
        'help': 'Output CSV used by NLP embed; defaults beside the input file.',
    }),
    CliOption(('--column-name', '--column_name'), {
        'default': 'content',
        'dest': 'column_name',
        'help': 'CSV text column used by NLP embed.',
    }),
)

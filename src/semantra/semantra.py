import hashlib
import io
import json
import math
import os
import sys
import gc
from pathlib import Path
from werkzeug.utils import secure_filename
import tempfile
import logging
import atexit
import signal
import click
import numpy as np
import pkg_resources
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, request, send_file, send_from_directory
from flask_cors import CORS
from tqdm import tqdm

from .models import BaseModel, TransformerModel, as_numpy, models
from .pdf import get_pdf_content
from .util import (
    HASH_LENGTH,
    collect_breakpoints,
    file_md5,
    get_annoy_filename,
    get_config_filename,
    get_embeddings_filename,
    get_faiss_filename,
    get_num_annoy_embeddings,
    get_num_faiss_embeddings,
    get_num_embeddings,
    get_offsets,
    get_tokens_filename,
    join_text_chunks,
    load_annoy_db,
    load_faiss_index,
    read_embeddings_file,
    sort_results,
    write_annoy_db,
    write_faiss_index,
    write_embedding,
)
from PyQt5.QtWidgets import QApplication, QFileDialog

VERSION = pkg_resources.require("semantra")[0].version
DEFAULT_ENCODING = "utf-8"
DEFAULT_PORT = 5000

package_directory = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_primary_query_text(queries):
    for query in queries or []:
        text = query.get("query")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def get_rerank_settings():
    model_name = os.getenv("OPENAI_RERANKING_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not model_name or not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        endpoint = f"{trimmed}/rerank"
    else:
        endpoint = f"{trimmed}/v1/rerank"

    return {
        "endpoint": endpoint,
        "model": model_name,
        "api_key": api_key,
    }


def rerank_sub_results(queries, sub_results):
    if not sub_results:
        return sub_results

    settings = get_rerank_settings()
    if settings is None:
        return sub_results

    query_text = get_primary_query_text(queries)
    if not query_text:
        return sub_results

    documents = [item.get("text", "") or "" for item in sub_results]
    payload = {
        "model": settings["model"],
        "query": query_text,
        "documents": documents,
    }
    headers = {
        "Authorization": f"Bearer {settings['api_key']}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            settings["endpoint"], json=payload, headers=headers, timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("Rerank request failed: %s", exc)
        return sub_results

    ordering = []
    scores_by_index = {}

    if isinstance(data, dict):
        candidates = data.get("data") or data.get("results") or data.get("items")
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                idx = (
                    item.get("document")
                    if isinstance(item.get("document"), int)
                    else item.get("index")
                )
                if idx is None or not isinstance(idx, int):
                    continue
                score = (
                    item.get("score")
                    if isinstance(item.get("score"), (int, float))
                    else item.get("relevance_score")
                )
                ordering.append(idx)
                if isinstance(score, (int, float)):
                    scores_by_index[idx] = float(score)

    if not ordering:
        return sub_results

    seen = set()
    reordered = []

    for idx in ordering:
        if idx < 0 or idx >= len(sub_results) or idx in seen:
            continue
        item = sub_results[idx]
        if idx in scores_by_index:
            item["rerank_score"] = scores_by_index[idx]
        reordered.append(item)
        seen.add(idx)

    # Append any leftover results that were not returned by the reranker
    for idx, item in enumerate(sub_results):
        if idx not in seen:
            reordered.append(item)

    return reordered


class Content:
    def __init__(self, rawtext, filename, filetype="text"):
        self.rawtext = rawtext
        self.filename = filename
        self.filetype = filetype


def get_text_content(md5, filename, semantra_dir, force, silent, encoding):
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return get_pdf_content(md5, filename, semantra_dir, force, silent)

    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise RuntimeError(
                "Excel support requires openpyxl. Please install the dependency."
            ) from exc

        workbook = load_workbook(filename=filename, read_only=True, data_only=True)
        try:
            text_lines = []
            for sheet in workbook.worksheets:
                text_lines.append(f"# Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    values = [str(cell) for cell in row if cell not in (None, "")]
                    if values:
                        text_lines.append("\t".join(values))
                text_lines.append("")
            rawtext = "\n".join(text_lines).strip()
        finally:
            workbook.close()

        return Content(rawtext, filename, filetype="spreadsheet")

    with open(filename, "r", encoding=encoding, errors="ignore") as f:
        rawtext = f.read()
        return Content(rawtext, filename)


TRANSFORMER_POOL_DEFAULT = 15000


class Document:
    def __init__(
        self,
        filename,
        md5,
        semantra_dir,
        base_filename,
        config,
        embeddings_filenames,
        vector_backend,
        annoy_filenames,
        faiss_filenames,
        windows,
        offsets,
        tokens_filename,
        num_dimensions,
        encoding,
    ):
        self.filename = filename
        self.md5 = md5
        self.semantra_dir = semantra_dir
        self.base_filename = base_filename
        self.config = config
        self.embeddings_filenames = embeddings_filenames
        self.annoy_filenames = annoy_filenames
        self.faiss_filenames = faiss_filenames
        self.vector_backend = vector_backend
        self.use_annoy = vector_backend == "annoy"
        self.use_faiss = vector_backend == "faiss"
        self._faiss_index = None
        self.windows = windows
        self.offsets = offsets
        self.tokens_filename = tokens_filename
        self.num_dimensions = num_dimensions
        self.encoding = encoding

    @property
    def content(self):
        return get_text_content(
            self.md5, self.filename, self.semantra_dir, False, True, self.encoding
        )

    @property
    def text_chunks(self):
        with open(self.tokens_filename, "r") as f:
            return json.loads(f.read())

    @property
    def num_embeddings(self):
        return len(self.offsets[0])

    @property
    def embedding_db(self):
        if not self.use_annoy:
            raise ValueError("Embeddings are not stored in Annoy database")
        return load_annoy_db(self.annoy_filenames[0], self.num_dimensions)

    @property
    def faiss_index(self):
        if not self.use_faiss:
            raise ValueError("Embeddings are not stored in Faiss index")
        if self._faiss_index is None:
            self._faiss_index = load_faiss_index(self.faiss_filenames[0])
        return self._faiss_index

    @property
    def embeddings(self):
        results, embedding_count = read_embeddings_file(
            self.embeddings_filenames[0],
            self.num_dimensions,
            self.num_embeddings,
        )
        assert embedding_count == self.num_embeddings
        return results


def process(
    filename,
    semantra_dir,
    model,
    num_dimensions,
    index_backend,
    num_annoy_trees,
    windows,
    cost_per_token,
    pool_count,
    pool_size,
    force,
    silent,
    no_confirm,
    encoding,
):
    # Check if semantra dir exists
    if not os.path.exists(semantra_dir):
        os.makedirs(semantra_dir)

    suffix = Path(filename).suffix.lower()
    use_annoy = index_backend == "annoy"
    use_faiss = index_backend == "faiss"

    # Get the md5 and config
    md5 = file_md5(filename)
    base_filename = os.path.basename(filename)
    config = model.get_config()
    if encoding != DEFAULT_ENCODING:
        config["encoding"] = encoding
    config_hash = hashlib.shake_256(json.dumps(config).encode()).hexdigest(HASH_LENGTH)

    # File names
    tokens_filename = os.path.join(semantra_dir, get_tokens_filename(md5, config_hash))
    config_filename = os.path.join(semantra_dir, get_config_filename(md5, config_hash))

    should_calculate_tokens = True
    if force or not os.path.exists(tokens_filename):
        # Calculate tokens to get text chunks
        content = get_text_content(md5, filename, semantra_dir, force, silent, encoding)
        text = content.rawtext
        tokens = model.get_tokens(text)
        should_calculate_tokens = False
        text_chunks = model.get_text_chunks(text, tokens)
        with open(tokens_filename, "w") as f:
            f.write(json.dumps(text_chunks))
    else:
        with open(tokens_filename, "r") as f:
            text_chunks = json.loads(f.read())
    num_tokens = len(text_chunks)

    force_breakpoints = suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}
    prefer_breakpoints = suffix in {".txt", ".pdf"}
    break_chars = ("\n",)
    if suffix == ".pdf":
        break_chars = ("\r\n", "\n", "\r", "\f")
    elif suffix == ".txt":
        break_chars = ("\r\n", "\n", "\r")

    breakpoints = (
        collect_breakpoints(text_chunks, break_chars)
        if (force_breakpoints or prefer_breakpoints)
        else None
    )

    # Get embedding offsets based on config parameters
    (
        offsets,
        num_embedding_tokens,
    ) = get_offsets(
        num_tokens,
        windows,
        breakpoints=breakpoints,
        prefer_breakpoints=prefer_breakpoints,
        force_breakpoints=force_breakpoints,
    )

    # Full config contains additional details
    full_config = {
        **config,
        "filename": filename,
        "md5": md5,
        "base_filename": base_filename,
        "num_dimensions": num_dimensions,
        "cost_per_token": cost_per_token,
        "windows": windows,
        "num_tokens": num_tokens,
        "num_embeddings": len(offsets),
        "num_embedding_tokens": num_embedding_tokens,
        "use_annoy": use_annoy,
        "use_faiss": use_faiss,
        "vector_backend": index_backend,
        "num_annoy_trees": num_annoy_trees,
        "semantra_version": VERSION,
    }

    if force or not os.path.exists(config_filename):
        if cost_per_token is not None and not no_confirm:
            click.confirm(
                f"Tokens will cost ${num_embedding_tokens * cost_per_token:.2f}. Proceed?",
                abort=True,
            )

    # Write out the config every time
    with open(config_filename, "w") as f:
        f.write(json.dumps(full_config))

    embeddings_filenames = []
    annoy_filenames = []
    faiss_filenames = []
    with tqdm(
        total=num_embedding_tokens,
        desc=f"Embedding {base_filename}",
        leave=False,
        disable=silent,
    ) as pbar:
        for (size, offset, rewind), sub_offsets in zip(windows, offsets):
            embeddings_filename = os.path.join(
                semantra_dir,
                get_embeddings_filename(md5, config_hash, size, offset, rewind),
            )
            annoy_filename = os.path.join(
                semantra_dir,
                get_annoy_filename(
                    md5, config_hash, size, offset, rewind, num_annoy_trees
                ),
            )
            faiss_filename = os.path.join(
                semantra_dir,
                get_faiss_filename(md5, config_hash, size, offset, rewind),
            )
            embeddings_filenames.append(embeddings_filename)
            annoy_filenames.append(annoy_filename)
            faiss_filenames.append(faiss_filename)

            if os.path.exists(embeddings_filename) and (
                (not use_annoy or os.path.exists(annoy_filename))
                and (not use_faiss or os.path.exists(faiss_filename))
            ):
                num_embeddings = get_num_embeddings(embeddings_filename, num_dimensions)
                if use_annoy:
                    num_annoy_embeddings = get_num_annoy_embeddings(
                        annoy_filename, num_dimensions
                    )
                if use_faiss:
                    num_faiss_embeddings = get_num_faiss_embeddings(faiss_filename)

                if (
                    not force
                    and num_embeddings == len(sub_offsets)
                    and (not use_annoy or num_annoy_embeddings == len(sub_offsets))
                    and (not use_faiss or num_faiss_embeddings == len(sub_offsets))
                ):
                    # Embedding is fully calculated
                    continue

            if should_calculate_tokens:
                tokens = model.get_tokens(join_text_chunks(text_chunks))
                should_calculate_tokens = False

            # Read embeddings if they exist
            embedding_index = 0
            if not force and os.path.exists(embeddings_filename):
                embeddings, embedding_index = read_embeddings_file(
                    embeddings_filename, num_dimensions, len(sub_offsets)
                )
            else:
                embeddings = np.empty(
                    (len(sub_offsets), num_dimensions), dtype=np.float32
                )
                embedding_index = 0

            num_skip = embedding_index
            iteration = 0

            # Write embeddings
            pool = []
            pool_token_count = 0

            with open(embeddings_filename, "ab") as f:

                def flush_pool():
                    nonlocal pool, pool_token_count, embeddings, embedding_index, f

                    if len(pool) > 0:
                        embedding_results = model.embed(tokens, pool)
                        # Call .cpu if embedding_results contains it
                        if hasattr(embedding_results, "cpu"):
                            embedding_results = embedding_results.cpu()
                        embeddings[embedding_index : embedding_index + len(pool)] = (
                            embedding_results
                        )
                        for embedding in embedding_results:
                            write_embedding(f, embedding, num_dimensions)
                        embedding_index += len(pool)
                        pool = []
                        pool_token_count = 0

                for offset in sub_offsets:
                    size = offset[1] - offset[0]

                    # Skip if already calculated
                    if iteration < num_skip:
                        iteration += 1
                        pbar.update(size)
                        continue

                    window_text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                    if len(window_text) == 0:
                        pbar.update(size)
                        continue

                    pool.append(offset)
                    pool_token_count += size
                    if (
                        pool_count is not None and len(pool) >= pool_count
                    ) or pool_token_count >= pool_size:
                        flush_pool()
                    pbar.update(size)

                flush_pool()

            # Write embeddings db
            if use_annoy:
                write_annoy_db(
                    filename=annoy_filename,
                    num_dimensions=num_dimensions,
                    embeddings=embeddings,
                    num_trees=num_annoy_trees,
                )
            if use_faiss:
                write_faiss_index(
                    filename=faiss_filename,
                    embeddings=embeddings,
                )

    return Document(
        filename=filename,
        md5=md5,
        semantra_dir=semantra_dir,
        base_filename=base_filename,
        config=full_config,
        embeddings_filenames=embeddings_filenames,
        vector_backend=index_backend,
        annoy_filenames=annoy_filenames,
        faiss_filenames=faiss_filenames,
        windows=windows,
        offsets=offsets,
        tokens_filename=tokens_filename,
        num_dimensions=num_dimensions,
        encoding=encoding,
    )


def process_windows(windows: str) -> "list[tuple[int, int, int]]":
    for window in windows.split(","):
        if "_" in window:
            # One or two occurrences?
            if window.count("_") == 1:
                size, offset = window.split("_")
                rewind = 0
            else:
                size, offset, rewind = window.split("_")
            yield int(size), int(offset), int(rewind)
        else:
            yield int(window), 0, 0


def ask_for_pdf_file():
    app = QApplication(sys.argv)
    pdf_path, _ = QFileDialog.getOpenFileName(
        None, "Select a PDF file", "", "PDF Files (*.pdf)"
    )

    if not pdf_path:
        print("Error: No file selected.")
        sys.exit(1)
    app.quit()
    return (os.path.relpath(pdf_path),)


@click.command()
@click.argument("filename", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--model",
    type=click.Choice(models.keys(), case_sensitive=True),
    default="mpnet",
    show_default=True,
    help="Preset model to use for embedding",
)
@click.option(
    "--encoding",
    type=str,
    default=DEFAULT_ENCODING,
    show_default=True,
    help="Encoding to use for reading text files",
)
@click.option(
    "--transformer-model",
    type=str,
    help="Custom Huggingface transformers model name to use for embedding",
)
@click.option(
    "--windows",
    type=str,
    default="1024_0_16",
    show_default=True,
    help='Embedding windows to extract. A comma-separated list of the format "size[_offset=0][_rewind=0]. A window with size 1024, offset 0, and rewind of 16 (1024_0_16) will embed the document in chunks of roughly 1024 tokens which partially overlap by 16. Only the first window is used for search.',
)
@click.option(
    "--no-server",
    is_flag=True,
    default=False,
    show_default=True,
    help="Do not start the UI server (only process)",
)
@click.option(
    "--port",
    type=int,
    default=DEFAULT_PORT,
    show_default=True,
    help="Port to use for embedding server",
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    show_default=True,
    help="Host to use for embedding server. Set to 0.0.0.0 to make the server available externally.",
)
@click.option(
    "--pool-size",
    type=int,
    default=None,
    help="Max number of embedding tokens to pool together in requests",
)
@click.option(
    "--pool-count",
    type=int,
    default=None,
    help="Max number of embeddings to pool together in requests",
)
@click.option(
    "--doc-token-pre",
    type=str,
    default=None,
    help="Token to prepend to each document in transformer models (default: None)",
)
@click.option(
    "--doc-token-post",
    type=str,
    default=None,
    help="Token to append to each document in transformer models (default: None)",
)
@click.option(
    "--query-token-pre",
    type=str,
    default=None,
    help="Token to prepend to each query in transformer models (default: None)",
)
@click.option(
    "--query-token-post",
    type=str,
    default=None,
    help="Token to append to each query in transformer models (default: None)",
)
@click.option(
    "--num-results",
    type=int,
    default=10,
    show_default=True,
    help="Number of results (neighbors) to retrieve per file for queries",
)
@click.option(
    "--index-backend",
    type=click.Choice(["faiss", "annoy", "exact"], case_sensitive=False),
    default="faiss",
    show_default=True,
    help="Vector index backend to use for retrieval (faiss, annoy, or exact).",
)
@click.option(
    "--num-annoy-trees",
    type=int,
    default=100,
    show_default=True,
    help="Number of trees to use for approximate kNN via Annoy",
)
@click.option(
    "--svm",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use SVM instead of any kind of kNN for queries (slower and only works on symmetric models)",
)
@click.option(
    "--svm-c",
    type=float,
    default=1.0,
    show_default=True,
    help="SVM regularization parameter; higher values penalize mispredictions more",
)
@click.option(
    "--explain-split-count",
    type=int,
    default=9,
    show_default=True,
    help="Number of splits on a given window to use for explaining a query",
)
@click.option(
    "--explain-split-divide",
    type=int,
    default=6,
    show_default=True,
    help="Factor to divide the window size by to get each split length for explaining a query",
)
@click.option(
    "--num-explain-highlights",
    type=int,
    default=2,
    show_default=True,
    help="Number of split results to highlight for explaining a query",
)
@click.option(
    "--force", is_flag=True, default=False, help="Force process even if cached"
)
@click.option(
    "--silent",
    is_flag=True,
    default=False,
    help="Do not print progress information",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    default=False,
    help="Do not show cost and ask for confirmation before processing with OpenAI",
)
@click.option(
    "--version",
    is_flag=True,
    default=False,
    help="Print version and exit",
)
@click.option(
    "--list-models",
    is_flag=True,
    default=False,
    help="List preset models and exit",
)
@click.option(
    "--show-semantra-dir",
    is_flag=True,
    default=False,
    help="Print the directory semantra will use to store processed files and exit",
)
@click.option(
    "--semantra-dir",
    type=click.Path(exists=False),
    default=None,
    help="Directory to store semantra files in",
)
@click.option(
    "--search",
    type=str,
    default=None,
    help="Search directly and either print the results, or save to a file using --search <QUERY> --save-search-to <PATH>",
)
@click.option(
    "--save-search-to",
    type=click.Path(exists=False, writable=True),
    default=None,
    help="Where to save the results of the direct search using --search <QUERY>",
)

@click.option(
    "-show-dialog",
    is_flag=True,
    default=False,
    help="Show file dialog to select a file when no files are specified.",
)


def main(
    filename,
    windows="1024_0_16",
    no_server=False,
    port=5000,
    host="0.0.0.0",
    pool_size=None,
    pool_count=None,
    doc_token_pre=None,
    doc_token_post=None,
    query_token_pre=None,
    query_token_post=None,
    model="mpnet",
    transformer_model=None,
    encoding=DEFAULT_ENCODING,
    num_annoy_trees=100,
    num_results=10,
    index_backend="faiss",
    svm=False,
    svm_c=1.0,
    explain_split_count=9,
    explain_split_divide=6,
    num_explain_highlights=2,
    force=False,
    silent=False,
    no_confirm=False,
    version=False,
    list_models=False,
    show_semantra_dir=False,
    semantra_dir=None,  # auto
    search=None,
    save_search_to=None,
    show_dialog=False,
):
    if version:
        print(VERSION)
        return

    if list_models:
        for model_name in models:
            print(f"- {model_name}")
        return

    if semantra_dir is None:
        semantra_dir = click.get_app_dir("Semantra")

    if show_semantra_dir:
        print(semantra_dir)
        return
    # Load environment from Semantra dir
    env_path = os.path.join(semantra_dir, ".env")
    load_dotenv(env_path)

    # Default to empty files list
    if filename is None or len(filename) == 0:
        # Show file dialog only if explicitly requested
        if show_dialog:
            try:
                filename = ask_for_pdf_file()
            except Exception as e:
                print(e)
                # Fall back to starting with no files instead of error
                print("Starting Semantra with no files loaded.")
                filename = ()
        else:
            print("Starting Semantra with no files loaded.")
            filename = ()  # Empty tuple

    if filename and len(filename) > 0:
        print(f"Opening Semantra with {filename}")
    else:
        print("Opening Semantra with no files")

    processed_windows = list(process_windows(windows))
    vector_backend = index_backend.lower()
    if vector_backend not in {"faiss", "annoy", "exact"}:
        raise ValueError("index-backend must be one of 'faiss', 'annoy', or 'exact'")
    use_annoy_backend = vector_backend == "annoy"
    use_faiss_backend = vector_backend == "faiss"

    if transformer_model is not None:
        # Handle custom transformers model
        if pool_size is None:
            pool_size = TRANSFORMER_POOL_DEFAULT

        cost_per_token = None
        model = TransformerModel(
            transformer_model,
            doc_token_pre=doc_token_pre,
            doc_token_post=doc_token_post,
            query_token_pre=query_token_pre,
            query_token_post=query_token_post,
        )
    else:
        # Pull preset model
        model_config = models[model]
        cost_per_token = model_config["cost_per_token"]
        if pool_size is None:
            pool_size = model_config["pool_size"]
        if pool_count is None:
            pool_count = model_config.get("pool_count", None)
        model: BaseModel = model_config["get_model"]()

    # Check if model is compatible
    if svm and model.is_asymmetric():
        raise ValueError(
            "SVM is not compatible with asymmetric models. "
            "Please use a symmetric model or kNN."
        )

    documents = {}
    pbar = tqdm(filename, disable=silent)
    for fn in pbar:
        pbar.set_description(f"{os.path.basename(fn)}")
        documents[fn] = process(
            filename=fn,
            semantra_dir=semantra_dir,
            model=model,
            num_dimensions=model.get_num_dimensions(),
            index_backend=vector_backend,
            num_annoy_trees=num_annoy_trees,
            windows=processed_windows,
            cost_per_token=cost_per_token,
            pool_count=pool_count,
            pool_size=pool_size,
            force=force,
            silent=silent,
            no_confirm=no_confirm,
            encoding=encoding,
        )

    cached_content = None
    cached_content_filename = None

    def get_content(filename):
        nonlocal cached_content, cached_content_filename
        # Check if we can pull from cache
        if filename == cached_content_filename:
            return cached_content
        # If not, grab content
        content = documents[filename].content
        # Cache the content
        cached_content_filename = filename
        cached_content = content
        # Return the now-cached content
        return content

    def cleanup_resources():
        print("Cleaning up resources before shutdown...")
        # Force garbage collection first to resolve any circular references
        gc.collect()

        # Close all document resources
        for doc_name, doc in list(documents.items()):
            try:
                if hasattr(doc, 'content'):
                    # Handle both PDFContent and Content classes
                    if hasattr(doc.content, 'close'):
                        doc.content.close()

                    # Explicitly set pdfium to None to avoid memory leaks
                    if hasattr(doc.content, 'pdfium'):
                        try:
                            if doc.content.pdfium is not None:
                                doc.content.pdfium.close()
                        except Exception:
                            pass
                        doc.content.pdfium = None
            except Exception as e:
                print(f"Error cleaning up document {doc_name}: {e}")

        # Force garbage collection again to clean up any newly dereferenced objects
        gc.collect()
        print("Resource cleanup completed")

    # Register the cleanup function to run when Python exits
    atexit.register(cleanup_resources)

    # Define the signal handler with proper scope for sys
    def signal_handler(sig, frame):
        print("\nServer shutting down...")
        cleanup_resources()
        # Use os._exit instead of sys.exit to avoid issues with Flask's reloader
        import os
        os._exit(0)

    # Register signal handlers where Flask app is created
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start a Flask server
    print("Starting flask server...")
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def base():
        return send_from_directory(
            pkg_resources.resource_filename("semantra", "client_public"),
            "index.html",
        )

    # Path for all the static files (compiled JS/CSS, etc.)
    @app.route("/<path:path>")
    def home(path):
        return send_from_directory(
            pkg_resources.resource_filename("semantra", "client_public"),
            path,
        )

    @app.route("/api/files", methods=["GET"])
    def api_files():
        """API endpoint to get a list of all available files"""
        try:
            files_list = [
                {
                    "basename": os.path.basename(doc.filename),
                    "filename": doc.filename,
                    "filetype": doc.content.filetype,
                }
                for doc in documents.values()
            ]
            return jsonify(files_list)
        except Exception as e:
            app.logger.error(f"Error in /api/files: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Replace or add the /api/upload endpoint in semantra.py
    @app.route("/api/upload", methods=["POST"])
    def upload_files():
        """API endpoint to handle file uploads directly"""
        if 'files' not in request.files:
            return jsonify({'error': 'No files in request'}), 400

        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # List to keep track of processed files
        processed_files = []

        # Create temporary directory for uploads if needed
        temp_dir = tempfile.mkdtemp()

        try:
            for file in files:
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)

                try:
                    app.logger.info(f"Processing file: {filename}")
                    # Process the file with existing semantra functionality
                    document = process(
                        filename=file_path,
                        semantra_dir=semantra_dir,
                        model=model,
                        num_dimensions=model.get_num_dimensions(),
                        index_backend=vector_backend,
                        num_annoy_trees=num_annoy_trees,
                        windows=processed_windows,
                        cost_per_token=cost_per_token,
                        pool_count=pool_count,
                        pool_size=pool_size,
                        force=False,
                        silent=True,
                        no_confirm=True,  # Don't ask for confirmation during API uploads
                        encoding=encoding,
                    )

                    # Add the file to documents dictionary
                    documents[file_path] = document

                    processed_files.append({
                        'basename': filename,
                        'filename': file_path,
                        'filetype': document.content.filetype,
                        'status': 'success'
                    })

                    app.logger.info(f"Successfully processed file: {filename}")

                except Exception as e:
                    app.logger.error(f"Error processing file {filename}: {str(e)}")
                    processed_files.append({
                        'basename': filename,
                        'filename': file_path,
                        'status': 'error',
                        'error': str(e)
                    })

            return jsonify({
                'status': 'success',
                'processed_files': processed_files
            })

        except Exception as e:
            app.logger.error(f"Error in /api/upload: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route("/api/delete", methods=["POST"])
    def delete_document():
        try:
            data = request.json
            if not data or 'filename' not in data:
                return jsonify({'error': 'No filename specified'}), 400

            filename = data['filename']
            logger.info(f"Request to delete document: {filename}")

            # Check if the file exists in our documents dictionary
            if filename not in documents:
                return jsonify({'error': 'File not found in index'}), 404

            # Close the PDF document properly
            document = documents[filename]
            if hasattr(document, 'content'):
                if hasattr(document.content, 'close'):
                    document.content.close()
                elif hasattr(document.content, 'pdfium') and document.content.pdfium is not None:
                    document.content.pdfium.close()

            # Remove the document from our documents dictionary
            del documents[filename]
            logger.info(f"Successfully deleted document: {filename}")

            return jsonify({
                'status': 'success',
                'message': f'Document {filename} deleted successfully'
            })

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route("/api/query", methods=["POST"])
    def query():
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        return jsonify(query_by_queries_and_preferences(queries, preferences))

    def query_by_search_term(search_term: str):
        queries = [
            {
                "query": search_term,
                "weight": 1,
            }
        ]
        preferences = []  # Since this is a fresh search
        return query_by_queries_and_preferences(queries, preferences)

    def query_by_queries_and_preferences(queries, preferences):
        if svm:
            svm_results = querysvm_by_queries_and_preferences(queries, preferences)
            return svm_results
        if use_annoy_backend:
            ann_results = queryann_by_queries_and_preferences(queries, preferences)
            return ann_results
        if use_faiss_backend:
            faiss_results = queryfaiss_by_queries_and_preferences(queries, preferences)
            return faiss_results

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            # Get kNN with cosine similarity
            distances = np.dot(embeddings, embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding)
            )
            sorted_ix = np.argsort(-distances)

            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            sub_results = []
            for index in sorted_ix[:num_results]:
                distance = float(distances[index])
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            sub_results = rerank_sub_results(queries, sub_results)
            results.append([doc.filename, sub_results])

        response = sort_results(results, True)
        return response

    @app.route("/api/querysvm", methods=["POST"])
    def querysvm():
        from sklearn import svm

        queries = request.json["queries"]
        preferences = request.json["preferences"]
        return jsonify(querysvm_by_queries_and_preferences(queries, preferences))

    def querysvm_by_queries_and_preferences(queries, preferences):
        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)
        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            x = np.concatenate([embeddings, embedding[None, ...]])
            y = np.zeros(len(embeddings) + 1)
            y[-1] = 1

            # Train the svm
            clf = svm.LinearSVC(
                class_weight="balanced",
                verbose=False,
                max_iter=10000,
                tol=1e-6,
                C=svm_c,
            )
            clf.fit(x, y)

            # Infer similarities
            similarities = clf.decision_function(x)[: len(embeddings)]
            sorted_ix = np.argsort(-similarities)

            text_chunks = doc.text_chunks
            offsets = doc.offsets
            sub_results = []
            for index in sorted_ix[:num_results]:
                distance = similarities[index]
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            sub_results = rerank_sub_results(queries, sub_results)
            results.append([doc.filename, sub_results])

        return sort_results(results, True)

    @app.route("/api/queryann", methods=["POST"])
    def queryann():
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        return jsonify(query_by_queries_and_preferences(queries, preferences))

    def queryann_by_queries_and_preferences(queries, preferences):

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        results = []
        for doc in documents.values():
            if not doc.use_annoy:
                continue
            embedding_db = doc.embedding_db
            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            sub_results = []
            for [index, distance] in zip(
                *embedding_db.get_nns_by_vector(embedding, num_results, -1, True)
            ):
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        # Convert distance from Euclidean distance of normalized vectors to cosine
                        "distance": 1 - distance**2.0 / 2.0,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            sub_results = rerank_sub_results(queries, sub_results)
            results.append([doc.filename, sub_results])
        return sort_results(results, True)

    def queryfaiss_by_queries_and_preferences(queries, preferences):
        import faiss  # type: ignore

        embedding = model.embed_queries_and_preferences(queries, preferences, documents)
        if embedding is None:
            return sort_results([], True)

        query_vec = np.array([embedding], dtype="float32")
        norms = np.linalg.norm(query_vec, axis=1)
        if np.any(norms == 0):
            return sort_results([], True)
        faiss.normalize_L2(query_vec)

        results = []
        for doc in documents.values():
            if not doc.use_faiss:
                continue
            index = doc.faiss_index
            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            distances, indices = index.search(query_vec, num_results)
            sub_results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(offsets):
                    continue
                offset = offsets[idx]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": float(distance),
                        "offset": offset,
                        "index": int(idx),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            sub_results = rerank_sub_results(queries, sub_results)
            results.append([doc.filename, sub_results])

        return sort_results(results, True)

    @app.route("/api/explain", methods=["POST"])
    def explain():
        filename = request.json["filename"]
        offset = request.json["offset"]
        tokens = documents[filename].text_chunks[offset[0] : offset[1]]
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        # Find hot-spots within the result tokens
        def get_splits(divide_factor=2, num_splits=3, start=0, end=len(tokens)):
            window_length = math.ceil((end - start) / divide_factor)
            split_length = math.ceil((end - start) / num_splits)
            splits = []
            for i in range(num_splits):
                splits.append(
                    (
                        start + i * split_length,
                        min(end, start + i * split_length + window_length),
                    )
                )
            return splits

        def exclude_window(start, end):
            nonlocal tokens
            return join_text_chunks(tokens[:start] + tokens[end:])

        def get_highest_ranked_split(splits):
            nonlocal tokens, embedding
            split_queries = [exclude_window(start, end) for start, end in splits]
            split_windows = np.array(
                [
                    as_numpy(model.embed_document(split_query))
                    for split_query in split_queries
                ]
            )
            distances = split_windows.dot(embedding) / (
                np.linalg.norm(split_windows, axis=1) * np.linalg.norm(embedding)
            )
            # Return the splits in order of highest to lowest ranked
            return sorted(zip(splits, distances), key=lambda x: x[1], reverse=False)

        def as_tokens(splits):
            nonlocal tokens
            indices = sorted([split[0] for split in splits], key=lambda x: x[0])
            last_index = 0
            chunks = []

            def append(start, end, type):
                if start >= end:
                    return
                nonlocal chunks, tokens
                chunks.append(
                    {
                        "text": join_text_chunks(tokens[start:end]),
                        "type": type,
                    }
                )

            for index in indices:
                append(last_index, index[0], "normal")
                append(max(index[0], last_index), index[1], "highlight")
                last_index = index[1]

            append(last_index, len(tokens), "normal")
            return chunks

        splits = get_splits(
            divide_factor=explain_split_divide,
            num_splits=explain_split_count,
            start=0,
            end=len(tokens),
        )
        top_splits = get_highest_ranked_split(splits)[:num_explain_highlights]
        return jsonify(as_tokens(top_splits))

    @app.route("/api/getfile", methods=["GET"])
    def getfile():
        filename = request.args.get("filename")
        content = get_content(filename)
        filename = content.filename
        return send_file(filename)

    @app.route("/api/pdfpositions", methods=["GET"])
    def pdfpositions():
        filename = request.args.get("filename")
        content = get_content(filename)
        if content.filetype == "pdf":
            return jsonify(content.positions)
        else:
            return jsonify([])

    @app.route("/api/pdfpage", methods=["GET"])
    def pdfpage():
        filename = request.args.get("filename")
        content = get_content(filename)
        page = request.args.get("page")
        scale = request.args.get("scale")
        if content.filetype == "pdf":
            pil_image = content.get_page_image_pil(int(page), float(scale))
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            response = make_response(img_byte_arr.getvalue())
            response.headers.set("Content-Type", "image/png")
            return response

    @app.route("/api/pdfchars", methods=["GET"])
    def pdfchars():
        filename = request.args.get("filename")
        content = get_content(filename)
        if content.filetype != "pdf":
            return jsonify([])
        page = request.args.get("page")
        return jsonify(content.get_page_chars(int(page)))

    @app.route("/api/text", methods=["GET"])
    def text():
        filename = request.args.get("filename")
        return jsonify(documents[filename].text_chunks)

    def save_dict_as_json_to_path(data: dict, path: str):
        full_path = Path(os.path.abspath(path))
        extension = os.path.splitext(full_path)[1]
        json_extension = ".json"
        is_a_json = extension == json_extension
        if not is_a_json:
            raise Exception(f"Can't save json to {full_path} as it is not a json file.")
        with open(full_path, "a") as json_file:
            json.dump(data, json_file)

    if search is not None:
        query_results = query_by_search_term(search)
        if save_search_to is not None:
            full_path = Path(os.path.abspath(save_search_to))
            save_dict_as_json_to_path(query_results, full_path)

        else:
            print(query_results)

    if not no_server:
        try:
            app.run(host=host, port=port, debug=True)
        except SystemExit as e:
            import sys

            sys.tracebacklimit = 0
            if port == DEFAULT_PORT:
                raise Exception(
                    f"Try running again and adding `--port <port>` to the command to specify a different port."
                ) from None
            else:
                raise Exception(
                    f"Try specifying a different port with `--port <port>`."
                ) from None


if __name__ == "__main__":
    main()

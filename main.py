import sys
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Optional

import click

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Semantra imports are local to the commands so that importing this module does
# not require the full dependency stack unless a command is executed.

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".html",
    ".htm",
    ".md",
    ".xlsx",
    ".xlsm",
    ".xltx",
    ".xltm",
}


def iter_dataset_files(datasets_dir: Path) -> Iterable[Path]:
    """Yield files under datasets_dir that Semantra knows how to ingest."""
    for path in sorted(datasets_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".ds_store":
            continue
        if suffix in SUPPORTED_EXTENSIONS or suffix == "":
            yield path


def resolve_files(datasets_dir: Path) -> List[Path]:
    """Return a list of supported files, raising if the directory is missing."""
    if not datasets_dir.exists():
        raise click.ClickException(f"Datasets directory {datasets_dir} does not exist.")
    files = list(iter_dataset_files(datasets_dir))
    if not files:
        raise click.ClickException(
            f"No supported files were found in {datasets_dir}. "
            "Supported extensions are: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    return files


@click.group()
def cli() -> None:
    """Utility CLI for working with Semantra inside this repository."""


@cli.command()
@click.option(
    "--datasets-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets"),
    help="Directory containing documents to index.",
)
@click.option(
    "--model",
    type=str,
    default="mpnet",
    help="Preset embedding model to use (must exist in semantra.models).",
)
@click.option(
    "--encoding",
    type=str,
    default="utf-8",
    help="File encoding used when reading plain-text files.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Rebuild embeddings even if cached files are present.",
)
@click.option(
    "--semantra-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory where Semantra stores its cache (defaults to the app dir).",
)
@click.option(
    "--windows",
    type=str,
    default="1024_0_16",
    help="Embedding windows configuration (matches Semantra CLI syntax).",
)
@click.option(
    "--index-backend",
    type=click.Choice(["faiss", "annoy", "exact"], case_sensitive=False),
    default="faiss",
    show_default=True,
    help="Vector index backend to use for retrieval.",
)
@click.option(
    "--num-annoy-trees",
    type=int,
    default=100,
    help="Number of trees for Annoy approximate nearest neighbour index (when using Annoy).",
)
@click.option(
    "--test",
    is_flag=True,
    default=False,
    help="Test mode: sample up to 3 files per extension for indexing.",
)
@click.option(
    "--jobs",
    type=int,
    default=1,
    show_default=True,
    help="Number of parallel workers powered by joblib (uses threading backend).",
)
def index(
    datasets_dir: Path,
    model: str,
    encoding: str,
    force: bool,
    semantra_dir: Optional[Path],
    windows: str,
    index_backend: str,
    num_annoy_trees: int,
    test: bool,
    jobs: int,
) -> None:
    """Pre-compute semantic indexes for every file in datasets."""
    files = resolve_files(datasets_dir)

    # Import lazily to avoid importing heavy dependencies unless required.
    from semantra import semantra as sem

    semantra_dir = semantra_dir or Path(click.get_app_dir("Semantra"))
    semantra_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_config = sem.models[model]
    except KeyError as exc:
        raise click.ClickException(f"Unknown model preset '{model}'.") from exc

    embedding_model = model_config["get_model"]()
    vector_backend = index_backend.lower()
    if vector_backend not in {"faiss", "annoy", "exact"}:
        raise click.ClickException("index-backend must be one of faiss, annoy, or exact")
    pool_size = model_config.get("pool_size")
    pool_count = model_config.get("pool_count")
    cost_per_token = model_config.get("cost_per_token")
    processed_windows = list(sem.process_windows(windows))

    if test:
        grouped = defaultdict(list)
        for path in files:
            grouped[path.suffix.lower()].append(path)
        sampled = []
        rng = random.Random()
        for suffix, paths in grouped.items():
            if not paths:
                continue
            count = min(3, len(paths))
            sampled.extend(rng.sample(paths, count))
        files = sorted(sampled)
        click.echo(
            f"Test mode enabled: sampled {len(files)} files across {len(grouped)} suffix groups."
        )

    if not files:
        click.echo("No files to index. Exiting.")
        return

    click.echo(
        f"Indexing {len(files)} files from {datasets_dir} "
        f"using model '{model}' (backend={vector_backend}, semantra_dir={semantra_dir})."
    )

    def process_path(path: Path, silent: bool) -> None:
        document = sem.process(
            filename=str(path),
            semantra_dir=str(semantra_dir),
            model=embedding_model,
            num_dimensions=embedding_model.get_num_dimensions(),
            index_backend=vector_backend,
            num_annoy_trees=num_annoy_trees,
            windows=processed_windows,
            cost_per_token=cost_per_token,
            pool_count=pool_count,
            pool_size=pool_size,
            force=force,
            silent=silent,
            no_confirm=True,
            encoding=encoding,
        )
        # Ensure any PDF handles are closed to avoid pypdfium warnings.
        try:
            content = document.content
            close_attr = getattr(content, "close", None)
            if callable(close_attr):
                close_attr()
        except Exception:
            # Closing is best-effort; ignore failures to avoid masking indexing success.
            pass

    total_files = len(files)
    if jobs <= 1 or total_files == 1:
        for index, path in enumerate(files, start=1):
            click.echo(f"[{index}/{total_files}] Indexing {path}...")
            process_path(path, silent=False)
            click.echo(f"[{index}/{total_files}] Finished {path}")
    else:
        from joblib import Parallel, delayed

        click.echo(f"Using joblib with {jobs} workers.")
        progress_lock = Lock()
        indexed_files = list(enumerate(files, start=1))
        with click.progressbar(length=total_files, label="Indexing files") as bar:

            def _task(index_path: tuple[int, Path]) -> None:
                idx, current_path = index_path
                with progress_lock:
                    click.echo(f"[{idx}/{total_files}] Indexing {current_path}...")
                process_path(current_path, silent=True)
                with progress_lock:
                    click.echo(f"[{idx}/{total_files}] Finished {current_path}")
                    bar.update(1)

            Parallel(n_jobs=jobs, backend="threading")(
                delayed(_task)(index_path) for index_path in indexed_files
            )

    click.echo("Indexing complete.")


@cli.command()
@click.option(
    "--datasets-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets"),
    help="Directory containing documents to load in the UI.",
)
@click.option("--host", type=str, default="0.0.0.0", help="Server host.")
@click.option("--port", type=int, default=5000, help="Server port.")
@click.option(
    "--encoding",
    type=str,
    default="utf-8",
    help="File encoding used when reading plain-text files.",
)
@click.option(
    "--model",
    type=str,
    default="mpnet",
    help="Preset embedding model to use (must exist in semantra.models).",
)
@click.option(
    "--semantra-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory containing pre-built Semantra indexes (required).",
)
@click.option(
    "--windows",
    type=str,
    default="1024_0_16",
    help="Embedding windows configuration (matches Semantra CLI syntax).",
)
@click.option(
    "--index-backend",
    type=click.Choice(["faiss", "annoy", "exact"], case_sensitive=False),
    default="faiss",
    show_default=True,
    help="Vector index backend to use when serving results.",
)
@click.option(
    "--num-annoy-trees",
    type=int,
    default=100,
    help="Number of trees for Annoy approximate nearest neighbour index (when using Annoy).",
)
def start(
    datasets_dir: Path,
    host: str,
    port: int,
    encoding: str,
    model: str,
    semantra_dir: Path,
    windows: str,
    index_backend: str,
    num_annoy_trees: int,
) -> None:
    """Launch the Semantra server using existing indexes only."""
    files = resolve_files(datasets_dir)
    semantra_dir = semantra_dir.expanduser().resolve()
    if not semantra_dir.exists():
        raise click.ClickException(
            f"Semantra cache directory '{semantra_dir}' does not exist. "
            "Run the index command first."
        )

    from semantra import semantra as sem
    from semantra.util import (
        HASH_LENGTH,
        file_md5,
        get_annoy_filename,
        get_faiss_filename,
        get_embeddings_filename,
        get_config_filename,
        get_tokens_filename,
    )

    try:
        model_config = sem.models[model]
    except KeyError as exc:
        raise click.ClickException(f"Unknown model preset '{model}'.") from exc

    embedding_model = model_config["get_model"]()
    config = embedding_model.get_config()
    if encoding != sem.DEFAULT_ENCODING:
        config["encoding"] = encoding
    config_hash = hashlib.shake_256(json.dumps(config).encode()).hexdigest(
        HASH_LENGTH
    )
    processed_windows = list(sem.process_windows(windows))
    vector_backend = index_backend.lower()
    if vector_backend not in {"faiss", "annoy", "exact"}:
        raise click.ClickException("index-backend must be one of faiss, annoy, or exact")
    use_annoy = vector_backend == "annoy"
    use_faiss = vector_backend == "faiss"

    indexed_files: list[Path] = []
    missing = set()
    for path in files:
        md5 = file_md5(str(path))
        token_path = semantra_dir / get_tokens_filename(md5, config_hash)
        config_path = semantra_dir / get_config_filename(md5, config_hash)
        if not token_path.exists() or not config_path.exists():
            missing.add(path)
            continue
        for size, offset, rewind in processed_windows:
            embeddings_path = semantra_dir / get_embeddings_filename(
                md5, config_hash, size, offset, rewind
            )
            if not embeddings_path.exists():
                missing.add(path)
                break
            if use_annoy:
                annoy_path = semantra_dir / get_annoy_filename(
                    md5, config_hash, size, offset, rewind, num_annoy_trees
                )
                if not annoy_path.exists():
                    missing.add(path)
                    break
            if use_faiss:
                faiss_path = semantra_dir / get_faiss_filename(
                    md5, config_hash, size, offset, rewind
                )
                if not faiss_path.exists():
                    missing.add(path)
                    break
        else:
            indexed_files.append(path)

    if not indexed_files:
        missing_list = "\n".join(f"- {path}" for path in list(missing)[:10])
        more = "" if len(missing) <= 10 else f"\n...and {len(missing) - 10} more."
        raise click.ClickException(
            "No indexed documents were found for the requested settings.\n"
            f"Checked directory: {datasets_dir}\n"
            f"Cache directory: {semantra_dir}\n"
            "Missing examples:\n"
            f"{missing_list}{more}\n"
            "Run the index command (without --test) to build indexes for these files."
        )

    if missing:
        click.echo(
            f"Skipping {len(missing)} files without matching caches. "
            f"Serving {len(indexed_files)} indexed files instead."
        )

    click.echo(
        f"Starting Semantra server on {host}:{port} with {len(indexed_files)} files (backend={vector_backend})..."
    )

    cli_args = [
        "--host",
        host,
        "--port",
        str(port),
        "--encoding",
        encoding,
        "--model",
        model,
        "--windows",
        windows,
        "--index-backend",
        vector_backend,
        "--semantra-dir",
        str(semantra_dir),
    ]
    if use_annoy:
        cli_args.extend(["--num-annoy-trees", str(num_annoy_trees)])
    cli_args.extend(str(path) for path in indexed_files)

    try:
        sem.main.main(cli_args, standalone_mode=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise click.ClickException(
                f"Semantra exited with status code {exc.code}"
            ) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        raise click.ClickException(f"Failed to start Semantra: {exc}") from exc


if __name__ == "__main__":
    cli(prog_name="semantra-cli")

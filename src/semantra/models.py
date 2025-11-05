import codecs
import os
from abc import ABC, abstractmethod

import numpy as np
import tiktoken
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

minilm_model_name = "sentence-transformers/all-MiniLM-L6-v2"
mpnet_model_name = "sentence-transformers/all-mpnet-base-v2"
sgpt_model_name = "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit"
sgpt_1_3B_model_name = "Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def filter_none(x):
    return [i for i in x if i is not None]


def as_numpy(x):
    # If x is a tensor, convert it to a numpy array
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


class BaseModel(ABC):
    @abstractmethod
    def get_num_dimensions(self) -> int:
        ...

    @abstractmethod
    def get_tokens(self, text: str):
        ...

    @abstractmethod
    def get_token_length(self, tokens) -> int:
        ...

    @abstractmethod
    def get_text_chunks(self, text: str, tokens) -> "list[str]":
        ...

    @abstractmethod
    def get_config(self):
        ...

    @abstractmethod
    def embed(self, tokens, offsets, is_query: bool = False) -> "list[list[float]]":
        ...

    def embed_document(self, document) -> "list[float]":
        tokens = self.get_tokens(document)
        return self.embed(tokens, [(0, self.get_token_length(tokens))], False)[0]

    def embed_query(self, query: str) -> "list[float]":
        tokens = self.get_tokens(query)
        return self.embed(tokens, [(0, self.get_token_length(tokens))], True)[0]

    def embed_queries(self, queries) -> "list[float]":
        all_embeddings = [
            as_numpy(self.embed_query(query["query"])) * query["weight"]
            for query in queries
        ]
        # Return sum of embeddings
        return np.sum(all_embeddings, axis=0)

    def embed_queries_and_preferences(self, queries, preferences, documents):
        query_embedding = self.embed_queries(queries) if len(queries) > 0 else None
        # Add preferences to embeddings
        return np.sum(
            [
                *([query_embedding] if query_embedding is not None else []),
                *[
                    documents[pref["file"]["filename"]].embeddings[
                        pref["searchResult"]["index"]
                    ]
                    * pref["weight"]
                    for pref in preferences
                ],
            ],
            axis=0,
        )

    def is_asymmetric(self):
        return False


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_name="text-embedding-ada-002",
        num_dimensions=1536,
        tokenizer_name="cl100k_base",
    ):
        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception(
                "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable or create a `.env` file with the key in the current working directory or the Semantra directory, which is revealed by running `semantra --show-semantra-dir`."
            )
        
        # Set custom base URL if provided (for OpenAI-compatible APIs)
        base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL") or os.getenv(
            "OPENAI_BASE_URL"
        )

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        organization = os.getenv("OPENAI_ORG_ID")
        if organization:
            client_kwargs["organization"] = organization

        self.client = OpenAI(**client_kwargs)

        env_model_name = os.getenv("OPENAI_EMBEDDING_MODEL")
        self.model_name = env_model_name if env_model_name else model_name

        self._default_num_dimensions = num_dimensions
        env_num_dimensions = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
        if env_num_dimensions:
            try:
                self._configured_num_dimensions = int(env_num_dimensions)
            except ValueError as exc:
                raise ValueError(
                    "OPENAI_EMBEDDING_DIMENSIONS must be an integer when provided."
                ) from exc
        else:
            self._configured_num_dimensions = None
        self.num_dimensions = None
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def get_config(self):
        return {
            "model_type": "openai",
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer.name,
        }

    def get_num_dimensions(self) -> int:
        self._ensure_num_dimensions()
        return self.num_dimensions

    def get_tokens(self, text: str):
        return self.tokenizer.encode(text)

    def get_token_length(self, tokens) -> int:
        return len(tokens)

    def get_text_chunks(self, _: str, tokens) -> "list[str]":
        decoder = codecs.getincrementaldecoder("utf-8")()
        return [
            decoder.decode(
                self.tokenizer.decode_single_token_bytes(token), final=False
            )
            for token in tokens
        ]

    def embed(self, tokens, offsets, _is_query=False) -> "list[list[float]]":
        texts = [tokens[i:j] for i, j in offsets]
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        embeddings = np.array([data.embedding for data in response.data])
        if self.num_dimensions is None or embeddings.shape[1] != self.num_dimensions:
            self.num_dimensions = embeddings.shape[1]
        return embeddings

    def _ensure_num_dimensions(self):
        if self.num_dimensions is not None:
            return
        try:
            response = self.client.embeddings.create(
                model=self.model_name, input=[" "]
            )
            self.num_dimensions = len(response.data[0].embedding)
        except Exception as exc:
            if self._configured_num_dimensions is not None:
                self.num_dimensions = self._configured_num_dimensions
                return
            self.num_dimensions = self._default_num_dimensions
            raise RuntimeError(
                "Failed to determine embedding dimensions automatically. "
                "Provide OPENAI_EMBEDDING_DIMENSIONS to proceed."
            ) from exc


def zero_if_none(x):
    return 0 if x is None else x


class TransformerModel(BaseModel):
    def __init__(
        self,
        model_name,
        doc_token_pre=None,
        doc_token_post=None,
        query_token_pre=None,
        query_token_post=None,
        asymmetric=False,
        cuda=None,
    ):
        if cuda is None:
            cuda = torch.cuda.is_available()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Get tokens
        self.pre_post_tokens = [
            doc_token_pre,
            doc_token_post,
            query_token_pre,
            query_token_post,
        ]
        self.doc_token_pre = (
            self.tokenizer.encode(doc_token_pre, add_special_tokens=False)
            if doc_token_pre
            else None
        )
        self.doc_token_post = (
            self.tokenizer.encode(doc_token_post, add_special_tokens=False)
            if doc_token_post
            else None
        )
        self.query_token_pre = (
            self.tokenizer.encode(query_token_pre, add_special_tokens=False)
            if query_token_pre
            else None
        )
        self.query_token_post = (
            self.tokenizer.encode(query_token_post, add_special_tokens=False)
            if query_token_post
            else None
        )

        self.asymmetric = asymmetric

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()

    def get_config(self):
        return {
            "model_type": "transformers",
            "model_name": self.model_name,
            "doc_token_pre": self.pre_post_tokens[0],
            "doc_token_post": self.pre_post_tokens[1],
            "query_token_pre": self.pre_post_tokens[2],
            "query_token_post": self.pre_post_tokens[3],
            "asymmetric": self.asymmetric,
        }

    def is_asymmetric(self):
        return self.asymmetric

    def get_num_dimensions(self) -> int:
        return int(self.model.config.hidden_size)

    def get_tokens(self, text: str):
        return self.tokenizer(
            text, return_offsets_mapping=True, verbose=False, return_tensors="pt"
        )

    def get_token_length(self, tokens) -> int:
        return len(tokens["input_ids"][0])

    def get_text_chunks(self, text: str, tokens) -> "list[str]":
        offsets = tokens["offset_mapping"][0]
        chunks = []
        prev_i = None
        prev_j = None
        for i, j in offsets:
            new_i = prev_j if i == j else i
            if prev_i is not None:
                chunks.append(text[prev_i:new_i])
            if prev_i is None:
                prev_i = 0
            elif new_i > prev_i:
                prev_i = new_i
            if prev_j is None:
                prev_j = j
            elif j > prev_j:
                prev_j = j
        chunks.append(text[0 if prev_i is None else prev_i :])
        return chunks

    def normalize_input_ids(self, input_ids, is_query):
        if self.query_token_pre is None and self.query_token_post is None:
            return input_ids
        else:
            token_pre = self.query_token_pre if is_query else self.doc_token_pre
            token_post = self.query_token_post if is_query else self.doc_token_post
            return torch.cat(
                filter_none(
                    [
                        torch.tensor(token_pre) if token_pre is not None else None,
                        input_ids,
                        torch.tensor(token_post) if token_post is not None else None,
                    ]
                )
            )

    def normalize_attention_mask(self, attention_mask, is_query):
        if self.query_token_pre is None and self.query_token_post is None:
            return attention_mask
        else:
            token_pre = self.query_token_pre if is_query else self.doc_token_pre
            token_post = self.query_token_post if is_query else self.doc_token_post
            return torch.cat(
                filter_none(
                    [
                        torch.ones(len(token_pre)) if token_pre is not None else None,
                        attention_mask,
                        torch.ones(len(token_post)) if token_post is not None else None,
                    ]
                )
            )

    def embed(self, tokens, offsets, is_query=False) -> "list[list[float]]":
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                self.normalize_input_ids(
                    tokens["input_ids"][0].index_select(0, torch.tensor(range(i, j))),
                    is_query,
                )
                for i, j in offsets
            ],
            batch_first=True,
            padding_value=zero_if_none(self.tokenizer.pad_token_id),
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [
                self.normalize_attention_mask(
                    tokens["attention_mask"][0].index_select(
                        0, torch.tensor(range(i, j))
                    ),
                    is_query,
                )
                for i, j in offsets
            ],
            batch_first=True,
            padding_value=0,
        )
        if self.cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        with torch.no_grad():
            model_output = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return mean_pooling(model_output, attention_mask)


models = {
    "openai": {
        "cost_per_token": 0.0004 / 1000,
        "pool_size": 50000,
        "pool_count": 2000,
        "get_model": lambda: OpenAIModel(
            model_name="text-embedding-ada-002",
            num_dimensions=1536,
            tokenizer_name="cl100k_base",
        ),
    },
    "minilm": {
        "cost_per_token": None,
        "pool_size": 50000,
        "get_model": lambda: TransformerModel(model_name=minilm_model_name),
    },
    "mpnet": {
        "cost_per_token": None,
        "pool_size": 15000,
        "get_model": lambda: TransformerModel(model_name=mpnet_model_name),
    },
    "sgpt": {
        "cost_per_token": None,
        "pool_size": 10000,
        "get_model": lambda: TransformerModel(
            model_name=sgpt_model_name,
            query_token_pre="[",
            query_token_post="]",
            doc_token_pre="{",
            doc_token_post="}",
            asymmetric=True,
        ),
    },
    "sgpt-1.3B": {
        "cost_per_token": None,
        "pool_size": 1000,
        "get_model": lambda: TransformerModel(
            model_name=sgpt_1_3B_model_name,
            query_token_pre="[",
            query_token_post="]",
            doc_token_pre="{",
            doc_token_post="}",
            asymmetric=True,
        ),
    },
}


def resolve_model(name: str):
    """Return a model configuration, treating unknown names as HF identifiers."""
    config = models.get(name)
    if config is not None:
        return config

    fallback_pool_size = models.get("mpnet", {}).get("pool_size", 15000)
    return {
        "cost_per_token": None,
        "pool_size": fallback_pool_size,
        "get_model": lambda model_name=name: TransformerModel(model_name=model_name),
    }

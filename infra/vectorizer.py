# infra/vectorizer.py
from __future__ import annotations
import re, math, json, pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Literal
from collections import Counter

# ---------------- Base interface ----------------
class BaseSparseVectorizer(ABC):
    def __init__(self, pkl_path: Path, language: Literal["en","ko","auto"]):
        if language not in ("en", "ko", "auto"):
            raise ValueError("language must be 'en', 'ko', or 'auto'")
        self.pkl_path = Path(pkl_path)
        self.language = language

    @abstractmethod
    def fit(self, texts: List[str]): ...
    @abstractmethod
    def transform(self, texts: List[str]) -> List[Dict[str, Any]]: ...
    @abstractmethod
    def save(self): ...
    @abstractmethod
    def load(self) -> bool: ...

# ---------------- Tokenizers ----------------
HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
WORD_RE   = re.compile(r"[A-Za-z0-9_]+")

def _en_tokenize(text: str, enable_bigrams: bool = True) -> List[str]:
    t = (text or "").strip().lower()
    if not t:
        return []
    ws = WORD_RE.findall(t)
    toks: List[str] = list(ws)
    if enable_bigrams and len(ws) >= 2:
        toks.extend([ws[i] + "_" + ws[i + 1] for i in range(len(ws) - 1)])
    return toks

def _ko_tokenize_bigram(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    toks: List[str] = []
    for raw in t.split():
        toks.append(raw)  # 원형 보존
        chars = [ch for ch in raw if HANGUL_RE.match(ch)]
        for i in range(len(chars) - 1):
            toks.append(chars[i] + chars[i + 1])
    return toks

def _ko_tokenize_kiwi(text: str) -> List[str]:
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        tokens: List[str] = []
        for sent in kiwi.split_into_sents(text or ""):
            for tok in kiwi.tokenize(sent.text, normalize_coda=True):
                tokens.append(tok.form)
        return tokens or _ko_tokenize_bigram(text)
    except Exception:
        # kiwipiepy 미설치/오류 시 자동 폴백
        return _ko_tokenize_bigram(text)

# ---------------- BM25 ----------------
class BM25SparseVectorizer(BaseSparseVectorizer):
    def __init__(
        self,
        pkl_path: Path,
        language: Literal["en","ko","auto"] = "en",
        *,
        ko_tokenizer: Literal["bigram","kiwi"] = "bigram",
        k1: float = 1.2,
        b: float = 0.75,
        min_df: int = 2,
        max_df_ratio: float = 0.9,
    ):
        super().__init__(pkl_path, language)
        self.k1 = float(k1)
        self.b = float(b)
        self.min_df = int(min_df)
        self.max_df_ratio = float(max_df_ratio)
        self.ko_tokenizer = ko_tokenizer

        self.N: int = 0
        self.avgdl: float = 0.0
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    # ---- tokenization ----
    def _tokenize(self, text: str) -> List[str]:
        if self.language == "en":
            return _en_tokenize(text)
        if self.language == "ko":
            return _ko_tokenize_kiwi(text) if self.ko_tokenizer == "kiwi" else _ko_tokenize_bigram(text)
        # auto: 한글 포함 여부로 분기
        has_hangul = bool(HANGUL_RE.search(text or ""))
        if has_hangul:
            return _ko_tokenize_kiwi(text) if self.ko_tokenizer == "kiwi" else _ko_tokenize_bigram(text)
        return _en_tokenize(text)

    # ---- train ----
    def fit(self, texts: List[str]):
        if self.language == "auto":
            raise ValueError("Training language must be 'en' or 'ko' (not 'auto').")

        N = 0
        total_doc_len = 0
        df = Counter()

        for t in texts:
            toks = self._tokenize(t)
            if not toks:
                continue
            N += 1
            total_doc_len += len(toks)
            for tok in set(toks):
                df[tok] += 1

        if N == 0:
            raise ValueError("No documents to fit.")

        avgdl = total_doc_len / N
        max_df = int(self.max_df_ratio * N)

        vocab: Dict[str, int] = {}
        idf: Dict[str, float] = {}
        for token, d in df.items():
            if d < self.min_df or d > max_df:
                continue
            idf_val = math.log((N - d + 0.5) / (d + 0.5) + 1.0)
            if idf_val > 0:
                idx = len(vocab)
                vocab[token] = idx
                idf[token] = idf_val

        self.N = N
        self.avgdl = avgdl
        self.vocab = vocab
        self.idf = idf

    # ---- infer ----
    def _bm25_vec(self, text: str) -> Dict[int, float]:
        toks = self._tokenize(text)
        if not toks:
            return {}
        tf = Counter(t for t in toks if t in self.vocab)
        dl = sum(tf.values())
        if dl == 0:
            return {}

        vec: Dict[int, float] = {}
        denom_norm = (1.0 - self.b) + self.b * (dl / self.avgdl if self.avgdl > 0 else 1.0)
        for t, f in tf.items():
            idx = self.vocab[t]
            idf = self.idf.get(t, 0.0)
            denom = f + self.k1 * denom_norm
            score = idf * (f * (self.k1 + 1.0)) / denom
            if score > 0:
                vec[idx] = float(score)
        return vec

    def transform(self, texts: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in texts:
            sparse_map = self._bm25_vec(t)
            if sparse_map:
                idxs = sorted(sparse_map.keys())
                vals = [sparse_map[i] for i in idxs]
            else:
                idxs, vals = [], []
            out.append({"indices": idxs, "values": vals})
        return out

    # ---- persistence ----
    def save(self):
        data = {
            "meta": {
                "N": self.N,
                "avgdl": self.avgdl,
                "k1": self.k1,
                "b": self.b,
                "min_df": self.min_df,
                "max_df_ratio": self.max_df_ratio,
                "language": self.language,
                "ko_tokenizer": self.ko_tokenizer,
                "format_version": 1,
            },
            "vocab": self.vocab,
            "idf": self.idf,
        }
        self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pkl_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        if not self.pkl_path.exists():
            return False
        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)
        meta = data.get("meta", {})
        file_lang = meta.get("language")

        if file_lang and self.language != "auto" and file_lang != self.language:
            raise ValueError(f"Language mismatch: file({file_lang}) != requested({self.language})")

        self.N = int(meta.get("N", 0))
        self.avgdl = float(meta.get("avgdl", 0.0))
        self.k1 = float(meta.get("k1", self.k1))
        self.b = float(meta.get("b", self.b))
        self.min_df = int(meta.get("min_df", self.min_df))
        self.max_df_ratio = float(meta.get("max_df_ratio", self.max_df_ratio))
        if self.language == "auto" and file_lang in ("en", "ko"):
            self.language = file_lang
        self.ko_tokenizer = meta.get("ko_tokenizer", self.ko_tokenizer)

        self.vocab = data.get("vocab", {})
        self.idf = data.get("idf", {})
        return True

# --------- helpers: JSONL I/O (train/util에서 사용) ---------
def _iter_jsonl_texts(
    jsonl_paths: Iterable[str],
    text_field: str = "context",
    limit: Optional[int] = None
) -> Iterable[str]:
    count = 0
    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if limit is not None and count >= limit:
                    return
                obj = json.loads(line)
                text = obj.get(text_field) or obj.get("text") or obj.get("context") or obj.get("body") or ""
                yield text
                count += 1

def fit_from_jsonls(jsonl_paths: List[str],
                    pkl_path: str,
                    language: Literal["en","ko"] = "en",
                    *,
                    ko_tokenizer: Literal["bigram","kiwi"] = "bigram",
                    k1: float = 1.2, b: float = 0.75,
                    min_df: int = 2, max_df_ratio: float = 0.9,
                    text_field: str = "context",
                    limit: Optional[int] = None) -> BM25SparseVectorizer:
    vec = BM25SparseVectorizer(
        pkl_path=Path(pkl_path),
        language=language,
        ko_tokenizer=ko_tokenizer,
        k1=k1, b=b, min_df=min_df, max_df_ratio=max_df_ratio
    )
    texts: List[str] = list(_iter_jsonl_texts(jsonl_paths, text_field=text_field, limit=limit))
    vec.fit(texts)
    vec.save()
    return vec

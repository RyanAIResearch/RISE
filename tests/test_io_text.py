import pytest

from conftest import small_config
from rise.pipeline import iter_jsonl_blocks
from rise.text import explode, format_sample, query_chunks
from rise.utils.io import count_jsonl_rows, iter_jsonl


def _write(tmp_path, lines):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_row_numbering_skips_blank_lines_consistently(tmp_path):
    path = _write(tmp_path, ['{"i": 0}', "", '{"i": 1}', "   ", '{"i": 2}', '{"i": 3}', '{"i": 4}'])
    assert [r["i"] for r in iter_jsonl(path)] == [0, 1, 2, 3, 4]
    assert count_jsonl_rows(path) == 5
    blocks = list(iter_jsonl_blocks(path, block_size=2))
    assert [(b, s, [r["i"] for r in rows]) for b, s, rows in blocks] == [(0, 0, [0, 1]), (1, 2, [2, 3]), (2, 4, [4])]
    only = list(iter_jsonl_blocks(path, block_size=2, wanted={1}))
    assert [(b, s, [r["i"] for r in rows]) for b, s, rows in only] == [(1, 2, [2, 3])]
    limited = list(iter_jsonl_blocks(path, block_size=2, limit=3))
    assert sum(len(r) for _, _, r in limited) == 3


def test_bad_json_fails_with_line_number(tmp_path):
    path = _write(tmp_path, ['{"i": 0}', "{broken", '{"i": 2}'])
    with pytest.raises(ValueError, match=":2: invalid JSON"):
        list(iter_jsonl(path))
    with pytest.raises(ValueError, match=":2: invalid JSON"):
        list(iter_jsonl_blocks(path, block_size=10))
    # a worker that does not own the broken block does not need to parse it
    assert [b for b, _, _ in iter_jsonl_blocks(path, block_size=1, wanted={2})] == [2]


def test_format_sample_priority():
    assert format_sample({"text": " a ", "prompt": "p"}) == "a"
    assert format_sample({"prompt": "Q:", "generation": " A"}) == "Q: A"
    assert format_sample({"instruction": "do", "input": "x", "output": "y"}) == "do\nx\ny"
    assert format_sample({"text": "", "label": 1}) == ""  # an empty text is still an (empty) row
    with pytest.raises(ValueError, match=r"needs `text`.*\['answer', 'question'\]"):
        format_sample({"question": "q", "answer": "a"})  # would silently become an empty row


def test_explode_windows_cover_long_samples():
    cfg = small_config(chunk_size=10, chunk_overlap=3, seq_len=16, min_seq_len=2)
    toks = list(range(25))
    chunks = explode(toks, cfg)
    assert [c[0] for c in chunks] == [0, 7, 14, 21] and chunks[-1] == [21, 22, 23, 24]
    assert explode(list(range(8)), cfg) == [list(range(8))]
    assert explode([5], cfg) == []
    nochunk = small_config(chunk_long_sequences=False, seq_len=16)
    assert explode(toks, nochunk) == [toks[:16]]


def test_query_chunks_prompt_masking():
    cfg = small_config(seq_len=16)
    assert query_chunks(list(range(12)), list(range(5)), cfg) == [(list(range(12)), 5)]
    assert query_chunks(list(range(12)), list(range(30)), cfg) == [(list(range(12)), 11)]  # clamped to L-1
    assert query_chunks(list(range(3)), None, cfg) == [([0, 1, 2], 0)]


def test_blocks_with_lazy_claims(tmp_path):
    path = _write(tmp_path, [f'{{"i": {i}}}' for i in range(7)])
    seen = []
    got = list(iter_jsonl_blocks(path, block_size=3, wanted=lambda b: (seen.append(b), b != 1)[1]))
    assert seen == [0, 1, 2] and [(b, [r["i"] for r in rows]) for b, _, rows in got] == [(0, [0, 1, 2]), (2, [6])]

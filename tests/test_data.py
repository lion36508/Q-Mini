import json
from qmini.data import extract_text, iter_text_records

def test_extract_chat_messages():
    text = extract_text({"messages": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Mass attracts mass."}]})
    assert "[USER]" in text and "Explain gravity." in text and "Mass attracts mass." in text

def test_jsonl_and_plain_text(tmp_path):
    j = tmp_path / "chat.jsonl"; j.write_text(json.dumps({"question": "Q?", "answer": "A!"}) + "\n", encoding="utf-8")
    t = tmp_path / "notes.txt"; t.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    records = list(iter_text_records([j, t]))
    assert any("Q?" in x and "A!" in x for x in records)
    assert any("First paragraph." in x for x in records) and any("Second paragraph." in x for x in records)

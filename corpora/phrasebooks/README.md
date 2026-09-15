# `corpora/phrasebooks/` — the phrase lists a biasing run sends

A phrase book names the vocabularies a run hands to sessions, and the weight to send them
at. `verbatim-bench invariance --phrases <file>` reads one.

```json
{
  "name": "domains-v1",
  "boost": 2.0,
  "lists": {"medical": ["metformin", "..."], "legal": ["certiorari", "..."]},
  "unrelated": ["brontosaurus", "..."]
}
```

- **`lists`** are the per-session vocabularies. Half the corpus carries one and half
  carries none, interleaved, so a biased row and an unbiased row sit in the same batch.
  A stream keeps the same list at every concurrency level — the gate's question is what
  changes when batch composition changes, so the other input is held fixed.
- **`boost`** is NeMo's boosting alpha, not Riva's `boost` scale; the two are not
  calibrated against each other and the server caps what it will apply. `domains-v1`
  ships at **1.0**, which is where `verbatim-bench rare-terms` put the knee on a B300:
  recall 0.911 bare to 0.945, fourteen false accepts, word error rate unmoved. It shipped
  at 2.0 for a day on a weaker reading; 2.0 buys one occurrence less than 1.0 and pays
  six times the false accepts (105) and 23 percent relative word error rate.
- **`unrelated`** is the negative control: words with no relation to the corpus. Any of
  them that appears in a biased transcript and not in the bare one is recorded, because
  a weight high enough to insert list words into audio that merely sounds like them is
  worth knowing about whatever the invariance verdict says.

The book's digest goes on the record. With biasing on a transcript is a function of the
audio, the checkpoint, the phrase list and its weight; a record naming only the first two
would have quietly weakened "same audio, same transcript" to "same audio, and whoever
last edited the phrase list".

The positive control does **not** come from a book. It is built per clip from that clip's
own reference transcript, because the question it answers — did a phrase list reach the
decoder at all — is best answered by boosting the words the model is most likely to have
got wrong.

`domains-v1.json` holds a medical and a legal list of twenty entries each, chosen for the
shapes that tokenise badly: drug names, ICD-10 codes (`E11.9`), and citations
(`410 U.S. 113`). None of them occurs in LibriSpeech, which is deliberate for the
invariance arm: there they are an input that differs per stream, not an accuracy claim.

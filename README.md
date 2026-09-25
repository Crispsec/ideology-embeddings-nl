# Ideology embeddings — probing Dutch parliamentary speech with RobBERT

## Thesis

Can politics/ideologies be embedded into a vector space? If speeches from different
parties land in distinguishable regions of a language model's embedding space,
that space can be used to visualise ideologies and to quantify where a piece of
text sits between them.

This repo tests that with a first, crude setup: Tweede Kamer speeches embedded
with a Dutch BERT, visualised with UMAP, and quantified by projecting onto
party-to-party direction vectors.

## Approach

A proof of concept. Can you recover **ideology direction vectors** from a Dutch
BERT's embedding space, using Tweede Kamer speeches as labelled data, and then
project unseen text onto those axes?

Weakly. On the speeches it was fitted on, the PVV↔GroenLinks-PvdA axis pulls
PVV partly away from the other parties, but the distributions overlap heavily,
and a good part of what the vectors encode is debate procedure rather than
ideology. On short test sentences the axes fail: all six score toward PVV. So
this is a probe of what is recoverable, not a measurement of where any party
stands. The [Limitations](#limitations) section is the
interesting part.

![Party clusters in embedding space](party_clusters.png)

## Method

| Phase | What happens |
|-------|--------------|
| 1A | Fetch plenary speeches from the [Tweede Kamer open data API](https://opendata.tweedekamer.nl/) (`Fractie` → `FractieZetel` → `Spreekbeurt`), parse the Verslag XML |
| 1A′ | Strip the `Name (Party):` header that opens every speech, so the label is not in the input |
| 1B | Embed each speech with [RobBERT 2023](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base), mean-pooling the last hidden state over non-padding tokens |
| 1C | PCA to 50 dims (77.8% variance retained), then UMAP to 2D — do parties cluster? |
| 2A | Direction vector per party pair: `mean(A) − mean(B)`, normalised |
| 2B | Compare TF-IDF between the 20 highest- and 20 lowest-scoring speeches on an axis, to see what it actually encodes |
| 3 | Embed arbitrary Dutch text, project onto the axes, report a z-score per axis |

The direction-vector step is the crude version of the approach in Zou et al.,
[*Representation Engineering*](https://arxiv.org/abs/2310.01405) (2023). They
use the first PCA component of within-class covariance, which is more robust
than a raw mean difference. Swapping that in is the obvious next step.

## Results

All numbers are **in-sample**: the direction vectors are fitted on the same 232
speeches they are then evaluated on. There is no held-out split, cross-validation
or baseline yet, and with 768 dimensions and 14–80 speeches per party, in-sample
separation is expected to look better than it is.

**Header removed, result mostly unchanged.** An earlier run fed each speech in
with its `Mevrouw X (Party):` header, so the party label was in the input. With
the header stripped, the 20 speeches scoring most PVV-like are 14 PVV (16 with
the header), and the 20 most GroenLinks-PvdA-like are 10 GroenLinks-PvdA (10
before). On SP↔VVD: 14/20 SP (13 before) and 11/20 VVD (11 before). Speaker
names (`jimmy`, `dijk`, `michon`, `derkzen`) and party names disappear from
the top vocabulary; the procedural vocabulary stays.

The 1D score distributions overlap heavily. PVV sits to the right of the other
parties, but they are not cleanly separated:

![PVV vs GroenLinks-PvdA score distribution](PVV_GroenLinks-PvdA_scores.png)

**Short test sentences all score toward PVV**, including ones that should go the
other way:

```
PVV ↔ GroenLinks-PvdA
  'Nederland is vol. We moeten de grenzen sluiten en de immigratie stoppen.'            z=+2.84
  'We investeren in klimaatbeleid en werken samen met Europa aan een duurzame toekomst.' z=+0.63
  'De zorg moet toegankelijk zijn voor iedereen, niet alleen voor mensen met geld.'      z=+1.80
  'Ondernemers moeten minder belasting betalen zodat de economie kan groeien.'           z=+1.43

'Wij staan voor een open samenleving, een sterk Europa en gelijke kansen voor iedereen.'
  PVV ↔ D66              z=+1.86
  PVV ↔ GroenLinks-PvdA  z=+2.03
  SP  ↔ VVD              z=+1.23
  CDA ↔ D66              z=+1.01

'De elite in Den Haag luistert niet naar gewone Nederlanders. Genoeg is genoeg.'
  PVV ↔ D66              z=+2.66
  PVV ↔ GroenLinks-PvdA  z=+2.25
  SP  ↔ VVD              z=+0.93
  CDA ↔ D66              z=+1.35
```

A likely cause, not yet tested: one-sentence inputs (~80 characters) are far
outside the training distribution (speech-genre text, median ~450 characters),
and raw mean-pooled RoBERTa vectors are anisotropic and are not centred before
projecting, so short texts get a systematic shift. The z-score against training
scores does not correct for that.

## Limitations

These are real and they matter more than the results above.

1. **The axes encode procedure, not only ideology.** The vocabulary most loaded
   on the "GroenLinks-PvdA" end of the PVV↔GL axis includes `commissiedebat`,
   `plenair`, `namens`, `brief`, `motie`. That is debate-genre vocabulary. The
   vector is partly separating *what kind of parliamentary moment this was* from
   *who was speaking*.
2. **Evaluation is in-sample only.** See Results. Nothing here shows the axes
   generalise to speeches or speakers they were not fitted on, and there is no
   bootstrap or split-half check of how stable the axes are.
3. **Retrieval is noisy.** The 20 speeches scoring most "GroenLinks-PvdA-like"
   include 6 VVD speeches. A clean axis would not do that.
4. **Short texts fail.** All six demo sentences score toward PVV (see Results).
5. **Small, recent, unbalanced sample.** The analysis uses 232 speeches: up to
   80 per party for six parties (GroenLinks-PvdA 80, PVV 48, SP 45, VVD 27,
   D66 18, CDA 14), sampled from 563 fetched speeches across 17 parties from
   roughly 30 recent plenary sessions.
6. **Mean-pooled sentence embeddings are a blunt instrument** for this. Layer
   choice is unexplored — political content may be more concentrated in
   intermediate layers than in the last one.

Treat the output as a demonstration of the technique, not as a measurement of
any party's position.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab ideology_embeddings.ipynb
```

Run the cells in order. Phase 1A hits the Tweede Kamer API and caches to
`speeches.csv`; that file is committed, so you can skip straight to Phase 1B.
Embedding the 232 sampled speeches takes a few minutes on CPU and is cached to
`embeddings.npy` (gitignored, regenerated on first run).

## Data

`speeches.csv` holds 563 speech excerpts labelled by party, pulled from the
Tweede Kamer's public open data API. The notebook strips the speaker header
from each speech on load. An earlier fetch decoded the UTF-8 XML as Latin-1
(`KrÃ¶ger` for `Kröger`); the committed CSV has been repaired and the fetch
code now sets the encoding explicitly. These are published proceedings of a
national parliament: public record by design, spoken on the floor, already
attributed to named members. No private or scraped personal data is involved.

## Next steps

- Proper repE direction vectors (within-class covariance PCA) instead of mean difference
- Regress out the procedural/genre component before computing axes
- Held-out party validation: does ChristenUnie project sensibly without being in the fit?
- Historical drift: 2015–2017 speeches vs 2022–2024, same axes
- Calibrate against [Kieskompas](https://www.kieskompas.nl/) positions using party manifestos

## AI assistance

Built with AI assistance (Claude). The research question, the choice of method
and the reading of the results are mine; Claude contributed code, plots and
documentation drafting. Every number and quoted output in this README comes
from the notebook in this repo, so you can check them by running it.

## Credits

- [RobBERT 2023](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base) — DTAI, KU Leuven
- [Representation Engineering](https://arxiv.org/abs/2310.01405) — Zou et al., 2023
- [Tweede Kamer Open Data](https://opendata.tweedekamer.nl/)

## Licence

Not yet chosen, so default copyright applies: all rights reserved. Open an issue
if you want to reuse any of it.

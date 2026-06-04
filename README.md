# Genericity Analysis Pipeline

Analysis and plotting code accompanying our paper on the degree of *generalization*
(genericity) in human- and machine-generated text.

The pipeline tags text at the clause level with genericity labels, computes a
weighted genericity score per text, and produces the comparisons and figures used
in the paper. The finished dataset is released separately; the text-generation
pipeline is not part of this release.

## Data

The accompanying dataset is archived on Zenodo:
<https://doi.org/10.5281/zenodo.19594487>

## Repository contents

| Path | Purpose |
|------|---------|
| `run_tagger.py` | Tag a CSV of texts with clause-level genericity labels (the Anecdotal Discourse Classifier). |
| `generalization_metric.py` | Compute genericity scores from tagged data and generate analysis CSVs + plots. |
| `plotting.py` | Visualization functions and shared styling used by `generalization_metric.py`. |
| `metric_validation.py` | Robustness checks for the metric (ablation, uniform, inverse-frequency, and random-weight baselines). |
| `rst/` | Standalone scripts for the RST/discourse-relation analyses reported in the paper. |
| `env/` | Pinned `requirements.txt` (pip) and `environment.yml` (conda). |

## Dependencies

### Python

```shell
pip install -r ./env/requirements.txt

# or with conda
conda env create -f ./env/environment.yml
```

Then download the spaCy model used by the clause tagger:

```shell
python -m spacy download en_core_web_sm
```

### Clause classifier

We use the classifier trained by Hemmatian (2021) as part of our pipeline. It is
included as a git submodule. Load it before running `run_tagger.py`:

```shell
# either clone recursively...
git clone --recurse-submodules <repo-url>

# ...or, if already cloned:
git submodule update --init --recursive
```

## Pipeline

### 1. Tag text — `run_tagger.py`

Applies the Anecdotal Discourse Classifier to every clause in the input text and
writes a tagged CSV.

- **Input:** CSV with a `text` column (a `generated_text` column is renamed to `text` automatically).
- **Output:** the input rows plus a `clause2labels` column, a list of `(clause_text, tag)` tuples. Saved gzip-compressed.

```shell
python run_tagger.py --input path/to/data.csv --output ./data/data_tagged.csv.gz

# sample a fraction of rows (useful for quick tests)
python run_tagger.py --input path/to/data.csv --sample-data 0.1
```

If `--output` is omitted, the result is written to `./data/{input_stem}_tagged.csv.gz`.

### 2. Score and analyze — `generalization_metric.py`

Computes the genericity score for each text and generates comparison tables and
figures.

```shell
# Analyze a single tagged file (groups by the "model" column by default)
python generalization_metric.py --input ./data/data_tagged.csv.gz

# Group/compare by a different categorical column (e.g. domain, genre)
python generalization_metric.py --input ./data/data_tagged.csv.gz --compare-col domain

# Disable bootstrap resampling / error bars (for independent single-sample groups)
python generalization_metric.py --input ./data/data_tagged.csv.gz --compare-col domain --no-bootstrap

# Save more/fewer of the top & bottom texts to CSV (default 100)
python generalization_metric.py --input ./data/data_tagged.csv.gz --n-files 200

# With no --input, analyzes all ./data/corpus-*tagged.csv files
python generalization_metric.py
```

**Options**

- `--input, -i` — path to a tagged CSV. If omitted, all `./data/corpus-*tagged.csv` files are processed.
- `--compare-col, -c` — categorical column to group by (e.g. `model`, `domain`, `genre`). If omitted, texts are only scored (no comparison/plots).
- `--n-files, -n` — number of most/least generic and most variable texts to save to CSV (default 100).
- `--no-bootstrap` — disable bootstrap std/error bars in plots.

Results are written to `./results/{input_stem}/` and figures to `./plots/{input_stem}/`,
split-aware where a single `split` value is present.

### 3. Validate the metric — `metric_validation.py`

Re-scores the dataset under alternative weighting schemes to examine the influence of the weighting scheme. We use it to compare our theoretically motivated weighting scheme with random, inverted and frequency-sensitive weighting:

- leave-one-out **ablation** of each tag,
- **uniform** weights (0.0 / 0.5 / 1.0),
- **inverse-frequency** weights,
- inverted **group** weights, and
- **random** weights over many trials.

Set the dataset(s) to validate via the `TEST_INPUT` list near the top of the file,
then run `python metric_validation.py`. Output is written to
`./results/metric_validation/`.

### RST / discourse analyses — `rst/`

The scripts under `rst/` reproduce the discourse-relation analyses in the paper
(relation frequencies, discourse markers, attribution/elaboration/joint trends,
heatmaps, etc.). They read intermediate JSON/CSV inputs from the working directory
and write figures and tables alongside them; run them from a directory containing
the corresponding inputs.

## The genericity metric

Each clause is assigned one tag by the classifier, and each tag has a fixed weight.
The text-level score is the weighted clause count normalized by the number of clauses:

```
score = Σ ( weight(tag_i) × count(tag_i) ) / total_clauses
```

The score ranges from 0.0 (very specific) to 1.0 (very generic). Weights are
defined in `TAG_WEIGHTS` in `generalization_metric.py`:

| Weight | Tags |
|--------|------|
| **1.0** | `GENERIC SENTENCE (STATIC / DYNAMIC / HABITUAL)` |
| **0.5** | `BOUNDED EVENT (GENERIC)`, `UNBOUNDED EVENT (GENERIC)`, `COERCED STATE (GENERIC)`, `PERFECT COERCED STATE (GENERIC)` |
| **0.0** | `BOUNDED/UNBOUNDED EVENT (SPECIFIC)`, `BASIC STATE`, `COERCED STATE (SPECIFIC)`, `GENERALIZING SENTENCE (DYNAMIC/STATIVE)`, `OTHER`, `IMPERATIVE`, `QUESTION` |

## Input data format

A CSV with at least:

- `text` — the document text,
- a categorical column to compare on (default `model`; e.g. `domain`, `genre`),
- one of `global_idx`, `local_idx`, or `unique_id` as the text identifier,
- optionally a `split` column and any other metadata.

If `global_idx` is absent it is derived from `split` + `local_idx`.

## Outputs

`generalization_metric.py` writes, per dataset:

- `results/{name}/` — `scored_texts.csv`, `{compare_col}_comparison.csv`, `text_comparison.csv`, most/least generic and variance tables, length correlations, and a summary.
- `plots/{name}/` — group comparison, score distributions, text scatter, high/low variance heatmaps, and length-relationship figures.

## Plot styling

`plotting.py` defines a `MODEL_STYLE_MAP` of consistent (colorblind-aware)
colors and markers for the models in the paper. Any value not listed there is
assigned automatically from a seaborn palette, so the plots work for arbitrary
comparison columns (domains, genres, custom models). Set `SHOW_PLOTS = True` in
`plotting.py` to display figures interactively instead of only saving them.


## Citation

If you use this code or the accompanying dataset, please cite our work.

Dataset:

```bibtex
@dataset{kirkegaard_fomsgaard_2026_19594487,
  author    = {Kirkegaard Fomsgaard, Søren and
               Pastor, Martial and
               Dias, Gael and
               Oostdijk, Nelleke},
  title     = {AUGFOX: An augmented dataset of argumentative
               texts written by humans and machines},
  month     = apr,
  year      = 2026,
  publisher = {Zenodo},
  version   = 1,
  doi       = {10.5281/zenodo.19594487},
  url       = {https://doi.org/10.5281/zenodo.19594487},
}
```

<!-- TODO: add bibtex citation when proceedings are published.
If using our code, please cite our paper:

```bibtex
@software{TODO,
  title   = {TODO},
  author  = {TODO},
  year    = {TODO},
  url     = {TODO}
}
```
-->

## License

This code is released under the MIT License — see [LICENSE](LICENSE).

Note that the clause classifier included as a submodule
(`Anecedotal_Discourse_Classifier_Multitext`, hosted as a
[Hugging Face Space](https://huggingface.co/spaces/BabakScrapes/Anecedotal_Discourse_Classifier_Multitext))
is the work of Hemmatian (2021) and is subject to its own terms.

## Acknowledgements

The clause-level genericity classifier used in this pipeline was trained by
Babak Hemmatian. If you use it, please cite:

```bibtex
@thesis{hemmatianTakingHighRoad2021,
  title = {Taking the {{High Road}}: {{A Big Data Investigation}} of {{Natural Discourse}} in the {{Emerging U}}.{{S}}. {{Consensus}} about {{Marijuana Legalization}}},
  shorttitle = {Taking the {{High Road}}},
  author = {Hemmatian, Babak},
  date = {2021-10-15}
}
```

This work was supported by funding from the Horizon Europe research and innovation
programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. Views and
opinions expressed are however those of the author(s) only and do not necessarily
reflect those of the European Union or the European Research Executive Agency (REA).
Neither the European Union nor the granting authority can be held responsible for them.

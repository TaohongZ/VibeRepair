ReInFixForConDefects
====================

This directory contains a function-level adaptation of ReInFix for
`ConDefects/dataset_2024`.

The static-analysis tool layer keeps the original ReInFix tool names and prompt
text, but replaces Joern queries with local single-file analysis:

- Python: `ast`-based function, call, assignment, return, import, and control-flow scans.
- Java: method-range extraction plus regular-expression scans for calls, imports,
  assignments, returns, classes, and control-flow lines.
- Example-patch retrieval: Flask service over `retrieval_base/embedded_cause_97k.csv`,
  with parquet or pickle caching for preloaded embeddings.

Run the retrieval server from the repository root:

```bash
python -m ReinFixForConDecfects.src.rag_search_server
```

Run repair for one program:

```bash
python -m ReinFixForConDecfects.src.react_sf_gen_patch --program-id 54865781
```

Useful options:

```bash
python -m ReinFixForConDecfects.src.react_sf_gen_patch --limit 5 --language Python
python -m ReinFixForConDecfects.src.react_sf_gen_patch --program-id 51771365 --skip-patches
python -m ReinFixForConDecfects.src.react_sf_gen_patch --output-root ReinFixForConDecfects/output
```

Outputs:

- `output/sf_solutions/*.json`: parsed root-cause and suggestion records.
- `output/sf_patches/*.json`: ReInFix-style generated full-function patches.
- `output/fix_code_results/*.json`: one JSON per generated patch, shaped for the
  existing ConDefects validation helper.
- `output/log/*.log`: per-program agent logs.

The analysis and patch prompts are copied from the original single-function
ReInFix implementation. The ConDefects adapter only changes how project context
and tool observations are supplied.


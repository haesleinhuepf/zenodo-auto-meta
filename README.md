# zenodo-auto-meta

[![PyPI](https://img.shields.io/pypi/v/zenodo-auto-meta)](https://pypi.org/project/zenodo-auto-meta/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**zenodo-auto-meta** is a pip-installable command-line tool that automatically
suggests keywords/tags for [Zenodo](https://zenodo.org/) records that are
missing them, using an OpenAI-compatible LLM (e.g.
[Ollama](https://ollama.ai/)).

## How it works

1. **Fetch** – Downloads all record metadata from a Zenodo community in pages
   of 10, accumulating a full list.  Results are cached in
   `cache_zenodo_auto_meta_<community>.yml` for up to one week so subsequent
   runs are fast.
2. **Separate** – Splits records into those *with* tags and those *without*.
3. **Few-shot examples** – Selects records that have both a description and
   tags, and builds an OpenAI-compatible chat history in which the *user*
   provides a description and the *assistant* returns the matching tag list.
4. **Predict** – Appends each tag-less (but described) record as a new user
   message and calls the LLM to suggest tags.
5. **Save** – Writes `proposed_tags.yml` with the link, description, and
   proposed tags for every predicted record.
6. **Curate** – You review and edit `proposed_tags.yml` in your favourite
   text editor.
7. **Update** – Run the tool again with `--update proposed_tags.yml` to push
   the curated tags back to Zenodo via its REST API.

## Installation

```bash
pip install zenodo-auto-meta
```

Or for development:

```bash
git clone https://github.com/haesleinhuepf/Aau.git
cd Aau
pip install -e ".[dev]"
```

## Prerequisites

* A running [Ollama](https://ollama.ai/) instance (default) **or** any
  OpenAI-compatible LLM endpoint.
* A Zenodo personal access token (only required when updating records).

## Usage

### Step 1 — Predict tags

```bash
zenodo-auto-meta <community>
```

Example (community slug `nfdi4bioimage`):

```bash
zenodo-auto-meta nfdi4biodiv nfdi4bioimage
```

This saves `proposed_tags.yml` in the current directory.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--force-refresh` | off | Re-fetch records even if the cache is fresh |
| `--model` | `llama3` | LLM model name (also via `ZENODO_AUTO_META_MODEL`) |
| `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible API URL (also via `ZENODO_AUTO_META_BASE_URL`) |
| `--api-key` | `ollama` | API key (also via `ZENODO_AUTO_META_API_KEY`) |
| `--max-examples` | `20` | Maximum few-shot examples to include |
| `--output` | `proposed_tags.yml` | Path for the output YAML file |
| `--zenodo-url` | `https://zenodo.org/api` | Zenodo API URL (also via `ZENODO_URL`) |

### Step 2 — Curate the YAML file

Open `proposed_tags_<communty>.yml` and review the suggested tags.  Each entry looks
like:

```yaml
- link: https://zenodo.org/record/1234567
  description: Plain-text description ...
  proposed_tags: 
  - microscopy
  - python
  - bioimaging
```

Remove entries you do not want to update, and edit the `proposed_tags` lists
as needed.

### Step 3 — Apply curated tags

```bash
export ZENODO_TOKEN=<your-zenodo-personal-access-token>
zenodo-auto-meta <community> --update proposed_tags_<communtity>.yml
```

> **Note** – Only records *owned by the authenticated user* can be updated via
> the Zenodo REST API. The tool performs an unlock → update → re-publish
> workflow for each record.

## Environment variables

| Variable | Description |
|----------|-------------|
| `ZENODO_TOKEN` | Zenodo personal access token |
| `ZENODO_URL` | Zenodo API base URL |
| `ZENODO_AUTO_META_MODEL` | LLM model name |
| `ZENODO_AUTO_META_BASE_URL` | LLM API base URL |
| `ZENODO_AUTO_META_API_KEY` | LLM API key |

## License

MIT — see [LICENSE](LICENSE).

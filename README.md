# LLM HTML Arena

A dependency-free local benchmark and recording bench for comparing two OpenAI-compatible chat models on the same one-shot HTML generation task.

The app streams each response, separates reasoning from final output when the provider exposes it, records token usage and timing, then replays the two runs side by side with runnable HTML previews.

## Features

- Works with OpenRouter, local gateways, and other OpenAI-compatible APIs.
- Accepts a custom base URL and explicit model ID for each side.
- Runs the requests sequentially in a selectable order to keep rate-limit behavior predictable.
- Streams reasoning and final output independently when supported.
- Uses provider-reported usage for exact token counts when available.
- Replays both responses on a synchronized timeline for screen recording.
- Extracts generated HTML and opens it in sandboxed preview frames.
- Saves requests, responses, HTML, metrics, and summary data for every run.
- Includes an offline Demo mode, so the interface can be tested without an API key.
- Uses only the Python standard library; there is nothing to install.

## Requirements

- Python 3.10 or newer
- A modern Chromium-, Firefox-, or Safari-based browser
- An API key for real model runs

## Quick start

Clone the repository, enter its directory, and run:

```bash
python app.py --open
```

On Windows, you can also use:

```powershell
.\start.ps1
```

The app opens at `http://127.0.0.1:8765`. Click **Demo** first to verify the full stream, replay, and preview workflow without spending API credits.

## Configure a provider

The API key is read from the process environment and is never sent to the browser. The repository includes [`.env.example`](.env.example) as a reference, but the app intentionally does not load `.env` files automatically.

PowerShell:

```powershell
$env:OPENAI_API_KEY = 'your-api-key'
python app.py --open
```

Bash or zsh:

```bash
export OPENAI_API_KEY='your-api-key'
python app.py --open
```

Then fill in the connection fields in the UI:

| Field | Example | Notes |
| --- | --- | --- |
| Base URL | `https://openrouter.ai/api/v1` | The OpenAI-compatible API root, without `/chat/completions` |
| Model A | `provider/model-a` | Exact model ID accepted by the provider |
| Model B | `provider/model-b` | Exact model ID accepted by the provider |
| API key variable | `OPENAI_API_KEY` | Name of the server-side environment variable |
| Max output tokens | blank | Leave blank to omit `max_tokens` and `max_completion_tokens` |

Both models use the shared base URL configured for the run. API keys stay server-side; the browser only receives a boolean indicating whether `OPENAI_API_KEY` is set.

### OpenRouter example

```powershell
$env:OPENAI_API_KEY = 'sk-or-...'
python app.py --open
```

Use `https://openrouter.ai/api/v1` as the base URL and paste the model IDs shown by OpenRouter into the two model fields.

## Token limits and usage

When **Max output tokens** is empty, the app does not add an output-token cap to the request. The provider or model may still enforce its own context or completion limit.

Token totals are taken from the final API `usage` object. If a provider does not return usage during streaming, the UI shows that exact usage is unavailable instead of estimating it locally.

## Compatible API behavior

The server sends requests to:

```text
POST {base_url}/chat/completions
```

with `stream: true` and `stream_options.include_usage: true`. It supports common streaming fields including:

- `choices[0].delta.content`
- `choices[0].delta.reasoning`
- `choices[0].delta.reasoning_content`
- `usage.prompt_tokens`
- `usage.completion_tokens`
- `usage.total_tokens`

## Recording a comparison

1. Put the same prompt in the shared prompt editor.
2. Enter the base URL and exact model ID for both sides.
3. Start the comparison and wait until both models finish.
4. Check the completion status, token counts, and generated previews.
5. Press **Replay** and record the synchronized timeline in OBS or another screen recorder.
6. Open **Artifacts** if you want to inspect or reuse the raw output.

The replay timeline is based on the original arrival time of every streamed chunk, scaled to a compact recording-friendly duration.

## Run artifacts

Every comparison creates a local directory under `runs/`:

```text
runs/<run-id>/
├── run.json
├── a/
│   ├── request.json
│   ├── reasoning.txt
│   ├── content.txt
│   ├── metrics.json
│   └── output.html
└── b/
    └── ...
```

Generated runs are excluded from Git. Request artifacts contain the endpoint and payload for reproducibility, but never the `Authorization` header or API-key value.

## Safety notes

Model-generated HTML is untrusted content. The app serves previews with a restrictive content security policy and displays them in sandboxed iframes. Keep the server bound to `127.0.0.1`, review generated code before reusing it, and never place credentials inside a benchmark prompt.

## Project structure

```text
.
├── app.py              # HTTP server, API client, streaming parser, artifact writer
├── prompt.txt          # Default benchmark prompt
├── start.ps1           # Windows launcher
└── static/
    └── index.html      # Complete browser UI
```

## License

Released under the [MIT License](LICENSE).

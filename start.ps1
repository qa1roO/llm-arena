$ErrorActionPreference = 'Stop'
$App = Join-Path $PSScriptRoot 'app.py'

if (-not $env:OPENAI_API_KEY -and -not $env:MOONSHOT_API_KEY) {
    Write-Host 'OPENAI_API_KEY is not set. Demo and endpoints without authentication will still work.' -ForegroundColor Yellow
    Write-Host 'For OpenRouter or another hosted provider, stop the server, set $env:OPENAI_API_KEY, and start again.' -ForegroundColor Yellow
}

python $App --open

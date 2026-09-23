$repo = "C:\Users\user\Desktop\hackaton\hack-54f1ae1b-astana-hackers"
$deliverables = "C:\Users\user\Documents\Codex\2026-09-23\agentic-ai-agentic-ai-24-48\outputs"

Copy-Item -LiteralPath (Join-Path $deliverables "weather_archive.py") -Destination (Join-Path $repo "weather_archive.py") -Force
Copy-Item -LiteralPath (Join-Path $deliverables "forecast_agent.py") -Destination (Join-Path $repo "forecast_agent.py") -Force
Copy-Item -LiteralPath (Join-Path $deliverables "requirements.txt") -Destination (Join-Path $repo "requirements.txt") -Force
Copy-Item -LiteralPath (Join-Path $deliverables "README.md") -Destination (Join-Path $repo "README.md") -Force

Set-Location $repo
& "C:\Python312\python.exe" -m pip install -r ".\requirements.txt"
& "C:\Python312\python.exe" -m py_compile ".\ml_baseline.py", ".\weather_archive.py", ".\forecast_agent.py"

Write-Host "Advanced ML files installed in $repo"

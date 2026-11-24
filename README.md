cd C:\RaceClipper
[System.IO.File]::WriteAllText("C:\RaceClipper\README.md", (Get-Clipboard -Raw), (New-Object System.Text.UTF8Encoding $false))
git add README.md
git commit -m "Fix README.md UTF-8 encoding, badges and v11 commands"
git push
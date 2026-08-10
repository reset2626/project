$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$zipPath = Get-ChildItem -LiteralPath (Join-Path $env:USERPROFILE 'Downloads') -Filter '*.zip' |
    Where-Object { $_.Name -like '* - *.zip' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$extractPath = Join-Path $root 'tmp_hong_thesis_archive'

if (-not (Test-Path -LiteralPath $extractPath) -or -not (Get-ChildItem -LiteralPath $extractPath -Force | Select-Object -First 1)) {
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractPath
}

$groups = [ordered]@{
    m1m9 = Get-ChildItem -LiteralPath (Join-Path $root 'notebooks\experiments\m1m9') -File -Recurse
    exploration = Get-ChildItem -LiteralPath (Join-Path $root 'notebooks\exploration') -File -Recurse
    papers = Get-ChildItem -LiteralPath (Join-Path $root 'docs\papers') -File -Recurse
    archive_code = Get-ChildItem -LiteralPath $extractPath -File -Recurse | Where-Object { $_.Extension -in '.py', '.ipynb' }
    archive_documents = Get-ChildItem -LiteralPath $extractPath -File -Recurse | Where-Object { $_.Extension -in '.pdf', '.pptx' }
}

foreach ($group in $groups.GetEnumerator()) {
    "[$($group.Key)]"
    $group.Value | ForEach-Object {
        $relative = $_.FullName.Substring($root.Length).TrimStart('\')
        "{0}`t{1}" -f $relative, $_.Length
    }
}

$Url = 'https://boshang9.wordpress.com/blog/'
$OutputRoot = Join-Path -Path (Get-Location) -ChildPath 'SavedMedia'
while ($true) {
    $Timestamp = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
    $SaveDir = Join-Path -Path $OutputRoot -ChildPath $Timestamp
    New-Item -Path $SaveDir -ItemType Directory -Force
    try {
        $Response = Invoke-WebRequest -Uri $Url -Method Get -ErrorAction Stop
        $Response.Content | Out-File -FilePath (Join-Path -Path $SaveDir -ChildPath "page_$Timestamp.html") -Encoding UTF8
        $MediaUrls = [regex]::Matches($Response.Content, 'src\s*=\s*"([^"]+\.(?:jpg|jpeg|png|gif|bmp|mp4|webm|ogg|wav|mp3))"', 'IgnoreCase') |
                     ForEach-Object { $_.Groups[1].Value } |
                     Sort-Object -Unique
        foreach ($Item in $MediaUrls) {
            $FullUri = if ($Item -match '^https?://') { $Item } else { (New-Object System.Uri($Url, $Item)).AbsoluteUri }
            $Filename = Split-Path -Path $FullUri -Leaf
            Invoke-WebRequest -Uri $FullUri -OutFile (Join-Path -Path $SaveDir -ChildPath $Filename) -ErrorAction SilentlyContinue
        }
    } catch {}
    Start-Sleep -Seconds 3600
}

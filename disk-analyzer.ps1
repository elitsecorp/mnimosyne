<#
.SYNOPSIS
    Disk Space Analyzer - Shows which folders occupy the most space
#>

param(
    [string]$Path = "C:\",
    [int]$Depth = 3,
    [int]$TopN = 20
)

function Get-FolderSize {
    param(
        [string]$FolderPath,
        [int]$MaxDepth,
        [int]$CurrentDepth = 0
    )
    
    $totalSize = 0
    $items = @()
    
    try {
        $dir = Get-Item $FolderPath -ErrorAction Stop
        if ($dir -isnot [System.IO.DirectoryInfo]) { return 0 }
        
        $files = Get-ChildItem -Path $FolderPath -File -ErrorAction SilentlyContinue | 
            Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue
        $totalSize = if ($files.Sum) { $files.Sum } else { 0 }
        
        if ($CurrentDepth -lt $MaxDepth) {
            $subdirs = Get-ChildItem -Path $FolderPath -Directory -ErrorAction SilentlyContinue
            foreach ($subdir in $subdirs) {
                $subSize = Get-FolderSize -FolderPath $subdir.FullName -MaxDepth $MaxDepth -CurrentDepth ($CurrentDepth + 1)
                $totalSize += $subSize
                $items += [PSCustomObject]@{
                    Path = $subdir.FullName
                    SizeBytes = $subSize
                    SizeGB = [math]::Round($subSize / 1GB, 2)
                    SizeMB = [math]::Round($subSize / 1MB, 2)
                }
            }
        }
    }
    catch {
        # Skip inaccessible folders
    }
    
    if ($CurrentDepth -eq 0) {
        return $items | Sort-Object SizeBytes -Descending | Select-Object -First $TopN
    }
    return $totalSize
}

function Format-Size {
    param([long]$Bytes)
    
    if ($Bytes -ge 1TB) { return "{0:N2} TB" -f ($Bytes / 1TB) }
    elseif ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
    elseif ($Bytes -ge 1MB) { return "{0:N2} MB" -f ($Bytes / 1MB) }
    elseif ($Bytes -ge 1KB) { return "{0:N2} KB" -f ($Bytes / 1KB) }
    else { return "$Bytes Bytes" }
}

# Main execution
Write-Host "`nDISK SPACE ANALYZER" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor DarkGray
Write-Host "Scanning: $Path" -ForegroundColor Yellow
Write-Host "Depth: $Depth levels" -ForegroundColor Yellow
Write-Host "`nCalculating folder sizes... (this may take a moment)`n" -ForegroundColor DarkGray

$folders = Get-FolderSize -FolderPath $Path -MaxDepth $Depth

Write-Host "`nTOP $TopN LARGEST FOLDERS" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor DarkGray
Write-Host ("Rank".PadRight(6)) -NoNewline -ForegroundColor DarkGray
Write-Host ("Size".PadRight(12)) -NoNewline -ForegroundColor White
Write-Host "Folder Path" -ForegroundColor White
Write-Host "-" * 70 -ForegroundColor DarkGray

$rank = 1
foreach ($folder in $folders) {
    $size = Format-Size -Bytes $folder.SizeBytes
    $color = if ($folder.SizeGB -gt 10) { "Red" } 
             elseif ($folder.SizeGB -gt 5) { "Yellow" }
             elseif ($folder.SizeGB -gt 1) { "Cyan" }
             else { "White" }
    
    Write-Host ("$rank.".PadRight(6)) -NoNewline -ForegroundColor DarkGray
    Write-Host ($size.PadRight(12)) -NoNewline -ForegroundColor $color
    Write-Host $folder.Path -ForegroundColor White
    $rank++
}

Write-Host "`n" + "=" * 70 -ForegroundColor DarkGray

# Summary
$totalUsed = ($folders | Measure-Object -Property SizeBytes -Sum).Sum
Write-Host "`nSUMMARY" -ForegroundColor Green
Write-Host "Total scanned: $(Format-Size -Bytes $totalUsed)" -ForegroundColor Cyan
Write-Host "Largest folder: $($folders[0].Path) ($(Format-Size -Bytes $folders[0].SizeBytes))" -ForegroundColor Yellow

# Drive info
Write-Host "`nDRIVE INFO" -ForegroundColor Green
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -gt 0 } | ForEach-Object {
    $used = Format-Size -Bytes $_.Used
    $free = Format-Size -Bytes $_.Free
    $total = Format-Size -Bytes ($_.Used + $_.Free)
    $percent = [math]::Round(($_.Used / ($_.Used + $_.Free)) * 100, 1)
    
    Write-Host "$($_.Name): $used used / $free free ($total total) - $percent% used" -ForegroundColor White
}

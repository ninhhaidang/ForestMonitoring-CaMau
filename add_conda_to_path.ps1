# Script to add Conda to User PATH
# Run this with: PowerShell -ExecutionPolicy Bypass -File add_conda_to_path.ps1

Write-Host "Adding Conda to PATH..." -ForegroundColor Green

# Conda paths to add
$condaPaths = @(
    "C:\tools\Anaconda3",
    "C:\tools\Anaconda3\Scripts",
    "C:\tools\Anaconda3\Library\bin",
    "C:\tools\Anaconda3\condabin"
)

# Get current user PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Check and add each path if not already present
$modified = $false
foreach ($path in $condaPaths) {
    if ($currentPath -notlike "*$path*") {
        Write-Host "Adding: $path" -ForegroundColor Yellow
        $currentPath = "$currentPath;$path"
        $modified = $true
    } else {
        Write-Host "Already exists: $path" -ForegroundColor Gray
    }
}

# Update PATH if modified
if ($modified) {
    [Environment]::SetEnvironmentVariable("Path", $currentPath, "User")
    Write-Host "`nPATH updated successfully!" -ForegroundColor Green
    Write-Host "`nIMPORTANT: Please close and reopen your terminal for changes to take effect." -ForegroundColor Cyan
} else {
    Write-Host "`nAll Conda paths already in PATH. No changes needed." -ForegroundColor Green
}

# Display current conda paths in PATH
Write-Host "`nConda paths in PATH:" -ForegroundColor Yellow
foreach ($path in $condaPaths) {
    if ($currentPath -like "*$path*") {
        Write-Host "  [OK] $path" -ForegroundColor Green
    } else {
        Write-Host "  [NOT FOUND] $path" -ForegroundColor Red
    }
}

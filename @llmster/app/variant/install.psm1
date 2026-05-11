$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest 

function Write-Info {
  param([string]$Message)
  if ($Global:PrintQuiet -eq 0) { Write-Host $Message }
}

function Write-VerboseMessage {
  param([string]$Message)
  if ($Global:PrintVerbose -eq 1) { Write-Host $Message }
}

function Write-WarningMessage {
  param([string]$Message)
  if ($Global:PrintQuiet -eq 0) { Write-Warning $Message }
}

function Stop-Install {
  param([string]$Message)
  if ($Global:PrintQuiet -eq 0) { Write-Error $Message }
  exit 1
}

function New-DirectoryIfMissing {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

function Invoke-NoThrow {
  param([ScriptBlock]$Action)
  try { & $Action } catch { }
}



# llmster config
$APP_NAME = 'llmster'
$APP_VERSION = '0.0.12-1'
$APP_VARIANT = 'full'
$ZIP_EXT = '.zip'
$CHECKSUM_EXT = '.sha512'
$ARTIFACT_DOWNLOAD_URL = 'llmster.lmstudio.ai/download'



# Read environment-driven flags with explicit defaults
if ($null -ne $env:LMS_PRINT_VERBOSE -and $env:LMS_PRINT_VERBOSE -ne '') {
  $Global:PrintVerbose = [int]$env:LMS_PRINT_VERBOSE
}
elseif ($null -ne $env:INSTALLER_PRINT_VERBOSE -and $env:INSTALLER_PRINT_VERBOSE -ne '') {
  $Global:PrintVerbose = [int]$env:INSTALLER_PRINT_VERBOSE
}
else {
  $Global:PrintVerbose = 0
}
if ($null -ne $env:LMS_PRINT_QUIET -and $env:LMS_PRINT_QUIET -ne '') {
  $Global:PrintQuiet = [int]$env:LMS_PRINT_QUIET
}
elseif ($null -ne $env:INSTALLER_PRINT_QUIET -and $env:INSTALLER_PRINT_QUIET -ne '') {
  $Global:PrintQuiet = [int]$env:INSTALLER_PRINT_QUIET
}
else {
  $Global:PrintQuiet = 0
}
if ($null -ne $env:LMS_NO_MODIFY_PATH -and $env:LMS_NO_MODIFY_PATH -ne '') {
  $Global:NoModifyPath = [int]$env:LMS_NO_MODIFY_PATH
}
elseif ($null -ne $env:INSTALLER_NO_MODIFY_PATH -and $env:INSTALLER_NO_MODIFY_PATH -ne '') {
  $Global:NoModifyPath = [int]$env:INSTALLER_NO_MODIFY_PATH
}
else {
  $Global:NoModifyPath = 0
}

function Get-HomeDirectory {
  if ($null -ne $env:HOME -and $env:HOME -ne '') { return $env:HOME }
  if ($null -ne $env:USERPROFILE -and $env:USERPROFILE -ne '') { return $env:USERPROFILE }
  return [Environment]::GetFolderPath('UserProfile')
}
$Global:InferredHome = Get-HomeDirectory

function Get-UsageText {
  @"
llmster-installer-$APP_VARIANT.ps1

The installer for LM Studio daemon (llmster) v$APP_VERSION

This script detects your Windows architecture, downloads the appropriate archive,
then unpacks and installs to:

  $HOME/.lmstudio/bin (or $env:LOCALAPPDATA/lm-studio/bin)

It will then attempt to add that dir to PATH by invoking `lms bootstrap -y`.

USAGE:
  llmster-installer-$APP_VARIANT.ps1 [OPTIONS]

OPTIONS:
  -v, --verbose
      Enable verbose output

  -q, --quiet
      Disable progress output

      --no-modify-path
      Don't configure the PATH environment variable (deprecated; use LMS_NO_MODIFY_PATH=1)

  -h, --help
      Print help information
"@
}

function Get-WindowsArchToken {
  if (($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') -or ($env:PROCESSOR_ARCHITEW6432 -eq 'ARM64')) {
    return 'arm64'
  }
  if (($env:PROCESSOR_ARCHITECTURE -eq 'AMD64') -or ($env:PROCESSOR_ARCHITEW6432 -eq 'AMD64')) {
    return 'x64'
  }
  Stop-Install "no compatible downloads were found for your platform win32/$($env:PROCESSOR_ARCHITECTURE)"
}

function Get-ReleaseName {
  $archToken = Get-WindowsArchToken
  return "$APP_VERSION-win32-$archToken.$APP_VARIANT"
}

function Resolve-Url {
  param([string]$Base)
  if ($Base.StartsWith('http://') -or $Base.StartsWith('https://')) { return $Base }
  return "https://$Base"
}

function Invoke-Downloader {
  param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$DestinationPath,
    [Parameter(Mandatory = $false)][int]$QuietOverride
  )
  $effectiveQuiet = if ($QuietOverride -ne $null) { [int]$QuietOverride } else { $Global:PrintQuiet }
  
  try {
    $normalized = Resolve-Url -Base $Url
    
    if ($effectiveQuiet -ne 1) {
      # Get file size first for progress calculation
      try {
        $headRequest = [System.Net.WebRequest]::Create($normalized)
        $headRequest.Method = "HEAD"
        $headResponse = $headRequest.GetResponse()
        $totalBytes = $headResponse.ContentLength
        $headResponse.Close()
      }
      catch {
        $totalBytes = 0
      }
      
      # Download with progress using HttpWebRequest
      $request = [System.Net.WebRequest]::Create($normalized)
      $response = $request.GetResponse()
      $responseStream = $response.GetResponseStream()
      $fileStream = [System.IO.File]::Create($DestinationPath)
      
      $buffer = New-Object byte[] 32768  # 32KB buffer
      $totalBytesRead = 0
      $lastProgressTime = [DateTime]::Now
      
      while ($true) {
        $bytesRead = $responseStream.Read($buffer, 0, $buffer.Length)
        if ($bytesRead -eq 0) { break }
        
        $fileStream.Write($buffer, 0, $bytesRead)
        $totalBytesRead += $bytesRead
        
        # Update progress every 250ms to avoid too frequent updates
        $now = [DateTime]::Now
        if (($now - $lastProgressTime).TotalMilliseconds -gt 250) {
          if ($totalBytes -gt 0) {
            $percent = [int](($totalBytesRead / $totalBytes) * 100)
            $mbReceived = [math]::Round($totalBytesRead / 1MB, 2)
            $mbTotal = [math]::Round($totalBytes / 1MB, 2)
            Write-Progress -Activity "Downloading $([System.IO.Path]::GetFileName($normalized))" -Status "$mbReceived MB / $mbTotal MB" -PercentComplete $percent
          }
          else {
            $mbReceived = [math]::Round($totalBytesRead / 1MB, 2)
            Write-Progress -Activity "Downloading $([System.IO.Path]::GetFileName($normalized))" -Status "$mbReceived MB downloaded" -PercentComplete -1
          }
          $lastProgressTime = $now
        }
      }
      
      Write-Progress -Activity "Downloading" -Completed
      $fileStream.Close()
      $responseStream.Close()
      $response.Close()
    }
    else {
      # Silent download - use the fast method without progress
      $webClient = New-Object System.Net.WebClient
      $webClient.DownloadFile($normalized, $DestinationPath)
      $webClient.Dispose()
    }
    
    return $true
  }
  catch {
    if ($fileStream) { $fileStream.Close() }
    if ($responseStream) { $responseStream.Close() }
    if ($response) { $response.Close() }
    if ($webClient) { $webClient.Dispose() }
    if ($effectiveQuiet -ne 1) {
      Write-Progress -Activity "Downloading" -Completed
    }
    return $false
  }
}

function Get-ChecksumOrEmpty {
  param([string]$ReleaseName, [string]$TempDir)
  $checksumFile = Join-Path $TempDir 'checksum.txt'
  # Prefer explicit <archive>.sha512; fallback to legacy .sha512
  $zipChecksumArtifact = "$ReleaseName$ZIP_EXT$CHECKSUM_EXT"
  $legacyChecksumArtifact = "$ReleaseName$CHECKSUM_EXT"
  $baseUrl = $ARTIFACT_DOWNLOAD_URL

  $preferredUrl = "$baseUrl/$zipChecksumArtifact"
  $legacyUrl = "$baseUrl/$legacyChecksumArtifact"

  if (Invoke-Downloader -Url $preferredUrl -DestinationPath $checksumFile -QuietOverride 1) {
    return (Get-Content -LiteralPath $checksumFile -Raw).Trim()
  }
  if (Invoke-Downloader -Url $legacyUrl -DestinationPath $checksumFile -QuietOverride 1) {
    return (Get-Content -LiteralPath $checksumFile -Raw).Trim()
  }
  return ''
}

function Test-Checksum {
  param([string]$FilePath, [string]$Checksum)
  if ($Checksum -eq '') { return }
  try {
    $hash = Get-FileHash -LiteralPath $FilePath -Algorithm SHA512
  }
  catch {
    Write-Info "Skipping checksum verification (missing Get-FileHash or file access error)"
    return
  }
  $calculated = $hash.Hash.ToLowerInvariant()
  $expected = $Checksum.ToLowerInvariant()
  if ($calculated -ne $expected) {
    Stop-Install "checksum mismatch`n  want: $expected`n  got:  $calculated"
  }
}

function Expand-ArchiveSafe {
  param([string]$ArchivePath, [string]$Destination)
  Expand-Archive -LiteralPath $ArchivePath -DestinationPath $Destination -Force
}

function Invoke-DownloadAndInstall {
  param([string[]]$Arguments)

  # Arg parsing
  foreach ($argument in $Arguments) {
    switch -Regex ($argument) {
      '^--help$' { Get-UsageText | Write-Host; exit 0 }
      '^--quiet$' { $Global:PrintQuiet = 1; continue }
      '^--verbose$' { $Global:PrintVerbose = 1; continue }
      '^--no-modify-path$' { Write-Info "--no-modify-path is deprecated; set LMS_NO_MODIFY_PATH=1"; $Global:NoModifyPath = 1; continue }
      '^-h$' { Get-UsageText | Write-Host; exit 0 }
      '^-v$' { $Global:PrintVerbose = 1; continue }
      '^-q$' { $Global:PrintQuiet = 1; continue }
      default { Stop-Install "unknown option $argument" }
    }
  }

  $releaseName = Get-ReleaseName
  $artifactName = "$releaseName$ZIP_EXT"
  $downloadBase = $ARTIFACT_DOWNLOAD_URL

  $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("llmster-install-" + [System.Guid]::NewGuid().ToString('N'))
  New-DirectoryIfMissing $tempRoot
  $archivePath = Join-Path $tempRoot ("input" + $ZIP_EXT)

  Write-Info "Downloading $APP_NAME $APP_VERSION Windows"
  Write-VerboseMessage "  from $downloadBase/$artifactName"
  Write-VerboseMessage "  to $archivePath"

  if (-not (Invoke-Downloader -Url "$downloadBase/$artifactName" -DestinationPath $archivePath)) {
    Write-Info "Failed to download $downloadBase/$artifactName."
    Write-Info "Please check your network connection."
    Write-Info ""
    Write-Info "If this problem persists, please feel free to open an issue"
    Write-Info "in https://github.com/lmstudio-ai/lmstudio-bug-tracker"
    exit 1
  }

  $checksumValue = Get-ChecksumOrEmpty -ReleaseName $releaseName -TempDir $tempRoot
  Test-Checksum -FilePath $archivePath -Checksum $checksumValue

  Write-Info "Installing... (this might take a few moments)"
  Expand-ArchiveSafe -ArchivePath $archivePath -Destination $tempRoot

  try {
    Install-Llmster -SourceDir $tempRoot
  }
  finally {
    Invoke-NoThrow { Remove-Item -LiteralPath $tempRoot -Recurse -Force }
  }
}



function Install-Llmster {
  param([Parameter(Mandatory = $true)][string]$SourceDir)

  $bootstrapExe = Join-Path $SourceDir 'llmster.exe'

  if (-not (Test-Path -LiteralPath $bootstrapExe)) {
    throw "Bootstrap binary not found at $bootstrapExe"
  }

  $previousBootstrapFlag = $env:LMS_BOOTSTRAP_INSTALL_SH
  $env:LMS_BOOTSTRAP_INSTALL_SH = '1'

  Write-Info 'Installing llmster...'
  & $bootstrapExe bootstrap
  $bootstrapExit = $LASTEXITCODE

  if ($null -ne $previousBootstrapFlag) {
    $env:LMS_BOOTSTRAP_INSTALL_SH = $previousBootstrapFlag
  }
  else {
    Remove-Item Env:LMS_BOOTSTRAP_INSTALL_SH -ErrorAction SilentlyContinue
  }

  if ($bootstrapExit -ne 0) {
    throw "llmster bootstrap failed with exit code $bootstrapExit"
  }

  $lmstudioHomePtr = Join-Path $Global:InferredHome '.lmstudio-home-pointer'
  $lmstudioHome = if (Test-Path -LiteralPath $lmstudioHomePtr) { Get-Content -LiteralPath $lmstudioHomePtr -Raw } else { Join-Path $Global:InferredHome '.lmstudio' }
  $lmsInstalledPath = Join-Path (Join-Path $lmstudioHome 'bin') 'lms.exe'

  if ($Global:NoModifyPath -eq 0 -and (Test-Path -LiteralPath $lmsInstalledPath)) {
    & $lmsInstalledPath bootstrap -y > $null 2>&1
    $bootstrapExit = $LASTEXITCODE
    if ($bootstrapExit -eq 0) {
      $lmsCommand = Get-Command 'lms' -ErrorAction SilentlyContinue
      $resolved = if ($null -ne $lmsCommand) { $lmsCommand.Path } else { $null }
      if ($resolved -ne $lmsInstalledPath) {
        Write-Info ''
        Write-Info 'To add lms to your PATH, either restart your shell or run:'
        Write-Info "    `$env:PATH = ""$($lmstudioHome)\bin;`$env:PATH"""
      }
    }
    else {
      Write-Info ''
      Write-Info 'Could not add lms to your PATH'
      Write-Info "lms install location: $lmsInstalledPath"
    }

    if ($null -ne $env:GITHUB_PATH -and $env:GITHUB_PATH -ne '') {
      Add-Content -LiteralPath $env:GITHUB_PATH -Value (Join-Path $lmstudioHome 'bin')
    }
  }
  elseif (Test-Path -LiteralPath $lmsInstalledPath) {
    Write-Info "lms install location: $lmsInstalledPath"
  }
}



Invoke-DownloadAndInstall $args

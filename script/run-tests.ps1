<#
.SYNOPSIS
    Runs GTest executables and generates a markdown test report.

.DESCRIPTION
    This script searches for GTest executables in a specified binary directory,
    runs them, and generates a comprehensive markdown report with test results.

.PARAMETER BinaryDirectory
    The directory containing the GTest executables.

.PARAMETER TestName
    The name pattern of the GTest executable(s). Supports wildcards (e.g., "*test*.exe").

.PARAMETER OutputReport
    Optional. The path to the output markdown report file. 
    Defaults to "test-report.md" in the current directory.

.PARAMETER FullTestOutput
    Optional. If specified, includes the full test output for failed tests instead of just error lines.
    Defaults to false (only error lines are included).

.PARAMETER ExcludeTests
    Optional. Pattern to exclude specific test executables. Supports wildcards (e.g., "*large_cases*").
    Test executables matching this pattern will be filtered out and not executed.

.EXAMPLE
    .\run-tests.ps1 -BinaryDirectory "C:\build\bin" -TestName "test_*.exe"

.EXAMPLE
    .\run-tests.ps1 -BinaryDirectory ".\build" -TestName "*test.exe" -OutputReport "test-results.md"

.EXAMPLE
    .\run-tests.ps1 -BinaryDirectory ".\build" -TestName "*test.exe" -FullTestOutput

.EXAMPLE
    .\run-tests.ps1 -BinaryDirectory ".\build" -TestName "*test.exe" -ExcludeTests "*large_cases*"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$BinaryDirectory,
    
    [Parameter(Mandatory=$true)]
    [string]$TestName,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputReport = "test-report.md",
    
    [Parameter(Mandatory=$false)]
    [switch]$FullTestOutput,
    
    [Parameter(Mandatory=$false)]
    [string]$ExcludeTests = ""
)

# Validate binary directory exists
if (-not (Test-Path -Path $BinaryDirectory -PathType Container)) {
    Write-Error "Binary directory does not exist: $BinaryDirectory"
    exit 1
}

# Find all matching executables
$executables = Get-ChildItem -Path $BinaryDirectory -Filter $TestName -File -Recurse -ErrorAction SilentlyContinue

# Filter out excluded executables if ExcludeTests is specified
if ($ExcludeTests) {
    $originalCount = $executables.Count
    $executables = $executables | Where-Object { $_.Name -notlike $ExcludeTests }
    $excludedCount = $originalCount - $executables.Count
    if ($excludedCount -gt 0) {
        Write-Host "Excluded $excludedCount executable(s) matching pattern '$ExcludeTests'"
    }
}

if ($executables.Count -eq 0) {
    Write-Error "No executables found matching pattern '$TestName' (after exclusions) in directory '$BinaryDirectory'"
    exit 1
}

Write-Host "Found $($executables.Count) executable(s) to run"

# Initialize counters
$totalTests = 0
$totalPassed = 0
$totalFailed = 0
$failedTestDetails = @()
$executionResults = @()

# Process each executable
foreach ($exe in $executables) {
    Write-Host "Running: $($exe.FullName)"
    
    $exeResult = @{
        Name = $exe.Name
        Path = $exe.FullName
        Tests = 0
        Passed = 0
        Failed = 0
        Output = ""
        FailedTests = @()
    }
    
    try {
        # Run the GTest executable
        $output = & $exe.FullName --gtest_color=no 2>&1 | Out-String
        $exeResult.Output = $output
        
        # Extract total tests run
        if ($output -match '\[==========\] Running (\d+) test') {
            $exeResult.Tests = [int]$matches[1]
            $totalTests += $exeResult.Tests
        }
        
        # Extract passed tests
        if ($output -match '\[  PASSED  \] (\d+) test') {
            $exeResult.Passed = [int]$matches[1]
            $totalPassed += $exeResult.Passed
        }
        
        # Extract failed tests count
        if ($output -match '\[  FAILED  \] (\d+) test') {
            $exeResult.Failed = [int]$matches[1]
            $totalFailed += $exeResult.Failed
        }
        
        $failedTestPattern = '\[  FAILED  \] ([^\r\n\(]+)'
        $failedMatches = [regex]::Matches($output, $failedTestPattern)
        
        foreach ($match in $failedMatches) {
            if ($match.Groups[1].Value -notmatch '^\d+ test') {
                $failedTestName = $match.Groups[1].Value.Trim()
                $exeResult.FailedTests += $failedTestName
                
                $parts = $failedTestName -split ", where "
                $escapedName = $parts[0]
                $runPattern = "\[\s+RUN\s+\]\s+$escapedName\s*[\r\n]+([\s\S]*?)\[\s+FAILED\s+\]\s+$escapedName.*"
                
                $detailsText = ""
                if ($output -match $runPattern) {
                    $testSection = $matches[1]
                    
                    if ($FullTestOutput) {
                        $detailsText = $testSection.Trim()
                    } else {
                        # Extract only lines containing "error" (case-insensitive)
                        $errorLines = @()
                        $lines = $testSection -split "`r?`n"
                        foreach ($line in $lines) {
                            if ($line -match 'error') {
                                $errorLines += $line.Trim()
                            }
                        }
                        
                        if ($errorLines.Count -gt 0) {
                            $detailsText = $errorLines -join "`n"
                        } else {
                            # If no error lines found, show the full section (might contain other useful info)
                            $detailsText = $testSection.Trim()
                            if ($detailsText.Length -lt 10) {
                                $detailsText = "Test failed without detailed error output."
                            }
                        }
                    }
                } else {
                    # If pattern doesn't match, provide a helpful message
                    $detailsText = "Test failed without detailed error output."
                }
                
                $failedTestDetails += @{
                    Executable = $exe.Name
                    TestName = $failedTestName
                    Details = $detailsText
                }
            }
        }
        
    } catch {
        Write-Warning "Error running $($exe.Name): $_"
        $exeResult.Output = "Error: $_"
    }
    
    $executionResults += $exeResult
}

# Generate Markdown Report
$reportContent = @"
# GTest Execution Report

**Generated:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Summary

| Metric | Count |
|--------|-------|
| **Total Tests Executed** | $totalTests |
| **Tests Passed** | $totalPassed |
| **Tests Failed** | $totalFailed |
| **Success Rate** | $(if ($totalTests -gt 0) { [math]::Round(($totalPassed / $totalTests) * 100, 2) } else { 0 })% |

## Executable Results

"@

foreach ($result in $executionResults) {
    # Use emoji/symbols via string concatenation to avoid encoding issues
    $passSymbol = [char]0x2705  # ✅ white heavy check mark
    $failSymbol = [char]0x274C  # ❌ cross mark
    $status = if ($result.Failed -eq 0) { "$passSymbol PASSED" } else { "$failSymbol FAILED" }
    
    $reportContent += "`n`n### $($result.Name) $status`n`n"
    $reportContent += "- **Tests Run:** $($result.Tests)`n"
    $reportContent += "- **Passed:** $($result.Passed)`n"
    $reportContent += "- **Failed:** $($result.Failed)`n"
    $reportContent += "- **Path:** ``$($result.Path)```n"
}

# Add failed test details section if there are failures
if ($failedTestDetails.Count -gt 0) {
    $failSymbol = [char]0x274C  # ❌ cross mark
    $reportContent += "`n`n---`n`n## Failed Test Details`n"

    foreach ($failure in $failedTestDetails) {
        $reportContent += "`n`n$failSymbol $($failure.TestName)`n`n"
        $reportContent += "$($failure.Details)`n"
    }
} else {
    $celebrationSymbol = [char]0x1F389 # 🎉 party popper 
    $reportContent += "`n`n---`n`n$celebrationSymbol All Tests Passed!`n`n"
    $reportContent += "No test failures detected.`n"
}

# Add footer
$reportContent += "`n"

# Write report to file
$reportContent | Out-File -FilePath $OutputReport -Encoding UTF8

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Execution Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total Tests:  $totalTests" -ForegroundColor White
Write-Host "Passed:       $totalPassed" -ForegroundColor Green
Write-Host "Failed:       $totalFailed" -ForegroundColor $(if ($totalFailed -gt 0) { "Red" } else { "Green" })
Write-Host "Report saved: $((Get-Item $OutputReport).FullName)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Exit with appropriate code
if ($totalFailed -gt 0) {
    exit 1
} else {
    exit 0
}

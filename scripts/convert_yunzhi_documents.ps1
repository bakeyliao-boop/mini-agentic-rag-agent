<#
这是 Windows + Microsoft Office COM 的可选数据准备脚本。

用途：将原始评测文档转换为 Source Markdown，并写入本机 knowledge/yunzhi-eval-v1。
它不是 multi-chunk-guide-001 Pilot 的运行依赖；跨平台运行 Pilot 时应直接使用
evaluation/fixtures 中已冻结的 Markdown 快照，不要在其他系统重新转换并覆盖该快照。
#>

param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$SemanticPathMappingPath = "",
    [string[]]$SampleId = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$datasetRoot = Join-Path $ProjectRoot "data\yunzhi-kb-eval"
$rawRoot = Join-Path $datasetRoot "raw"
$sourceManifestPath = Join-Path $datasetRoot "manifest.jsonl"
$outputRoot = Join-Path $ProjectRoot "knowledge\yunzhi-eval-v1"
$conversionManifestPath = Join-Path $datasetRoot "conversion-manifest.jsonl"
$defaultSemanticPathMappingPath = Join-Path $ProjectRoot "evaluation\yunzhi_document_semantic_paths.json"
if ([string]::IsNullOrWhiteSpace($SemanticPathMappingPath)) {
    $SemanticPathMappingPath = $defaultSemanticPathMappingPath
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function ConvertTo-YamlScalar {
    param([AllowEmptyString()][string]$Value)

    $escaped = $Value.Replace("\", "\\").Replace('"', '\"')
    $escaped = $escaped.Replace("`r", "").Replace("`n", "\n")
    return '"' + $escaped + '"'
}

function ConvertTo-CleanText {
    param([AllowNull()][object]$Value)

    if ($null -eq $Value) {
        return ""
    }
    $text = [string]$Value
    $text = $text.Replace([string][char]7, "")
    $text = $text.Replace([string][char]11, "`n").Replace([string][char]12, "`n")
    $text = $text.Replace("`r`n", "`n").Replace("`r", "`n")
    $lines = @($text.Split("`n") | ForEach-Object { ($_ -replace "\s+", " ").Trim() })
    return ($lines -join "`n").Trim()
}

function ConvertTo-MarkdownCell {
    param([AllowNull()][object]$Value)

    $text = ConvertTo-CleanText $Value
    return $text.Replace("\", "\\").Replace("|", "\|").Replace("`n", "<br>")
}

function Get-ColumnName {
    param([int]$Number)

    $name = ""
    $value = $Number
    while ($value -gt 0) {
        $value--
        $name = [char][int](65 + ([int]$value % 26)) + $name
        $value = [int][math]::Floor([double]$value / 26)
    }
    return $name
}

function New-SourceMarkdown {
    param(
        [string]$Title,
        [string]$FileFormat,
        [string]$SampleId,
        [string]$ContentLocatorType,
        [string[]]$BodyLines
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("---")
    $lines.Add("title: $(ConvertTo-YamlScalar $Title)")
    $lines.Add('resource_type: "doc"')
    $lines.Add("file_format: $(ConvertTo-YamlScalar $FileFormat)")
    $lines.Add("sample_id: $(ConvertTo-YamlScalar $SampleId)")
    $lines.Add("content_locator_type: $(ConvertTo-YamlScalar $ContentLocatorType)")
    $lines.Add("---")
    $lines.Add("")
    $lines.Add("# $Title")
    $lines.Add("")
    foreach ($line in $BodyLines) {
        $lines.Add($line)
    }
    while ($lines.Count -gt 0 -and [string]::IsNullOrWhiteSpace($lines[$lines.Count - 1])) {
        $lines.RemoveAt($lines.Count - 1)
    }
    return ($lines -join "`n") + "`n"
}

function Get-HeadingLevel {
    param([object]$Paragraph)

    try {
        $outlineLevel = [int]$Paragraph.OutlineLevel
        if ($outlineLevel -ge 1 -and $outlineLevel -le 5) {
            return $outlineLevel + 1
        }
    } catch {
        return 0
    }
    return 0
}

function Convert-WordTableToMarkdown {
    param([object]$Table)

    $rows = New-Object System.Collections.Generic.List[object]
    $maxColumns = 0
    foreach ($row in $Table.Rows) {
        $cells = New-Object System.Collections.Generic.List[string]
        foreach ($cell in $row.Cells) {
            $cells.Add((ConvertTo-MarkdownCell $cell.Range.Text))
        }
        $maxColumns = [math]::Max($maxColumns, $cells.Count)
        $rows.Add($cells.ToArray())
    }
    if ($rows.Count -eq 0 -or $maxColumns -eq 0) {
        return @()
    }

    $result = New-Object System.Collections.Generic.List[string]
    foreach ($row in $rows) {
        $values = New-Object System.Collections.Generic.List[string]
        for ($index = 0; $index -lt $maxColumns; $index++) {
            $values.Add($(if ($index -lt $row.Count) { $row[$index] } else { "" }))
        }
        $result.Add("| " + ($values -join " | ") + " |")
        if ($result.Count -eq 1) {
            $result.Add("| " + ((1..$maxColumns | ForEach-Object { "---" }) -join " | ") + " |")
        }
    }
    return $result.ToArray()
}

function Convert-WordLikeDocument {
    param(
        [object]$WordApplication,
        [string]$SourcePath,
        [bool]$IncludePageMarkers
    )

    $document = $null
    try {
        $document = $WordApplication.Documents.Open($SourcePath, $false, $true, $false)
        $blocks = New-Object System.Collections.Generic.List[object]

        foreach ($paragraph in $document.Paragraphs) {
            $insideTable = $false
            try { $insideTable = [bool]$paragraph.Range.Information(12) } catch { $insideTable = $false }
            if ($insideTable) {
                continue
            }
            $text = ConvertTo-CleanText $paragraph.Range.Text
            if ([string]::IsNullOrWhiteSpace($text)) {
                continue
            }
            $page = 0
            if ($IncludePageMarkers) {
                try { $page = [int]$paragraph.Range.Information(3) } catch { $page = 0 }
            }
            $blocks.Add([pscustomobject]@{
                Start = [int]$paragraph.Range.Start
                Page = $page
                Kind = "paragraph"
                Level = Get-HeadingLevel $paragraph
                Lines = @($text.Split("`n"))
            })
        }

        foreach ($table in $document.Tables) {
            $page = 0
            if ($IncludePageMarkers) {
                try { $page = [int]$table.Range.Information(3) } catch { $page = 0 }
            }
            $tableLines = @(Convert-WordTableToMarkdown $table)
            if ($tableLines.Count -gt 0) {
                $blocks.Add([pscustomobject]@{
                    Start = [int]$table.Range.Start
                    Page = $page
                    Kind = "table"
                    Level = 0
                    Lines = $tableLines
                })
            }
        }

        $result = New-Object System.Collections.Generic.List[string]
        $currentPage = 0
        foreach ($block in $blocks | Sort-Object Start) {
            if ($IncludePageMarkers -and $block.Page -gt 0 -and $block.Page -ne $currentPage) {
                if ($result.Count -gt 0) { $result.Add("") }
                $result.Add("<!-- page: $($block.Page) -->")
                $result.Add("")
                $currentPage = $block.Page
            }
            if ($block.Kind -eq "paragraph" -and $block.Level -gt 0 -and $block.Lines.Count -eq 1) {
                $result.Add(("#" * [math]::Min(6, $block.Level)) + " " + $block.Lines[0])
            } else {
                foreach ($line in $block.Lines) {
                    $result.Add($line)
                }
            }
            $result.Add("")
        }
        return $result.ToArray()
    } finally {
        if ($null -ne $document) {
            $document.Close(0)
        }
    }
}

function Convert-ExcelWorkbook {
    param(
        [object]$ExcelApplication,
        [string]$SourcePath
    )

    $workbook = $null
    try {
        $workbook = $ExcelApplication.Workbooks.Open($SourcePath, 0, $true)
        $result = New-Object System.Collections.Generic.List[string]
        foreach ($sheet in $workbook.Worksheets) {
            $result.Add("## Sheet: $($sheet.Name)")
            $result.Add("")
            $used = $sheet.UsedRange
            $rowCount = [int]$used.Rows.Count
            $columnCount = [int]$used.Columns.Count
            $startRow = [int]$used.Row
            $startColumn = [int]$used.Column
            $nonEmptyRows = New-Object System.Collections.Generic.List[object]
            $lastNonEmptyColumn = 0

            for ($rowIndex = 1; $rowIndex -le $rowCount; $rowIndex++) {
                $cells = New-Object System.Collections.Generic.List[string]
                $rowHasValue = $false
                for ($columnIndex = 1; $columnIndex -le $columnCount; $columnIndex++) {
                    $value = ConvertTo-MarkdownCell $used.Cells.Item($rowIndex, $columnIndex).Text
                    if (-not [string]::IsNullOrWhiteSpace($value)) {
                        $rowHasValue = $true
                        $lastNonEmptyColumn = [math]::Max($lastNonEmptyColumn, $columnIndex)
                    }
                    $cells.Add($value)
                }
                if ($rowHasValue) {
                    $nonEmptyRows.Add([pscustomobject]@{
                        Number = $startRow + $rowIndex - 1
                        Cells = $cells.ToArray()
                    })
                }
            }

            if ($nonEmptyRows.Count -eq 0 -or $lastNonEmptyColumn -eq 0) {
                $result.Add("_Empty worksheet_"
                )
                $result.Add("")
                continue
            }

            $endRow = $nonEmptyRows[$nonEmptyRows.Count - 1].Number
            $endColumnNumber = $startColumn + $lastNonEmptyColumn - 1
            $startCell = "$(Get-ColumnName $startColumn)$startRow"
            $endCell = "$(Get-ColumnName $endColumnNumber)$endRow"
            $result.Add("<!-- sheet: $(ConvertTo-YamlScalar ([string]$sheet.Name)) range: ${startCell}:${endCell} -->")
            $result.Add("")

            $headers = New-Object System.Collections.Generic.List[string]
            $headers.Add("row")
            for ($columnIndex = 0; $columnIndex -lt $lastNonEmptyColumn; $columnIndex++) {
                $headers.Add((Get-ColumnName ($startColumn + $columnIndex)))
            }
            $result.Add("| " + ($headers -join " | ") + " |")
            $result.Add("| " + ((1..$headers.Count | ForEach-Object { "---" }) -join " | ") + " |")
            foreach ($row in $nonEmptyRows) {
                $values = New-Object System.Collections.Generic.List[string]
                $values.Add([string]$row.Number)
                for ($columnIndex = 0; $columnIndex -lt $lastNonEmptyColumn; $columnIndex++) {
                    $values.Add($(if ($columnIndex -lt $row.Cells.Count) { $row.Cells[$columnIndex] } else { "" }))
                }
                $result.Add("| " + ($values -join " | ") + " |")
            }
            $result.Add("")
        }
        return $result.ToArray()
    } finally {
        if ($null -ne $workbook) {
            $workbook.Close($false)
        }
    }
}

function Convert-PowerPointTableToMarkdown {
    param([object]$Table)

    $result = New-Object System.Collections.Generic.List[string]
    for ($row = 1; $row -le $Table.Rows.Count; $row++) {
        $values = New-Object System.Collections.Generic.List[string]
        for ($column = 1; $column -le $Table.Columns.Count; $column++) {
            $values.Add((ConvertTo-MarkdownCell $Table.Cell($row, $column).Shape.TextFrame.TextRange.Text))
        }
        $result.Add("| " + ($values -join " | ") + " |")
        if ($row -eq 1) {
            $result.Add("| " + ((1..$Table.Columns.Count | ForEach-Object { "---" }) -join " | ") + " |")
        }
    }
    return $result.ToArray()
}

function Add-PowerPointShapeContent {
    param(
        [object]$Shape,
        [System.Collections.Generic.List[object]]$Blocks,
        [int]$TitleShapeId
    )

    if ([int]$Shape.Id -eq $TitleShapeId) {
        return
    }
    if ([int]$Shape.Type -eq 6) {
        for ($index = 1; $index -le $Shape.GroupItems.Count; $index++) {
            Add-PowerPointShapeContent $Shape.GroupItems.Item($index) $Blocks $TitleShapeId
        }
        return
    }
    try {
        if ([bool]$Shape.HasTable) {
            $lines = @(Convert-PowerPointTableToMarkdown $Shape.Table)
            if ($lines.Count -gt 0) {
                $Blocks.Add([pscustomobject]@{ Top = [double]$Shape.Top; Left = [double]$Shape.Left; Lines = $lines })
            }
            return
        }
    } catch {
        # The shape is not a table.
    }
    try {
        if ([bool]$Shape.HasTextFrame -and [bool]$Shape.TextFrame.HasText) {
            $text = ConvertTo-CleanText $Shape.TextFrame.TextRange.Text
            if (-not [string]::IsNullOrWhiteSpace($text)) {
                $Blocks.Add([pscustomobject]@{
                    Top = [double]$Shape.Top
                    Left = [double]$Shape.Left
                    Lines = @($text.Split("`n"))
                })
            }
        }
    } catch {
        # Ignore non-text shapes.
    }
}

function Convert-PowerPointPresentation {
    param(
        [object]$PowerPointApplication,
        [string]$SourcePath
    )

    $presentation = $null
    try {
        $presentation = $PowerPointApplication.Presentations.Open($SourcePath, $true, $false, $false)
        $result = New-Object System.Collections.Generic.List[string]
        foreach ($slide in $presentation.Slides) {
            $title = ""
            $titleShapeId = -1
            try {
                $titleShape = $slide.Shapes.Title
                if ($null -ne $titleShape -and [bool]$titleShape.HasTextFrame -and [bool]$titleShape.TextFrame.HasText) {
                    $title = ConvertTo-CleanText $titleShape.TextFrame.TextRange.Text
                    $titleShapeId = [int]$titleShape.Id
                }
            } catch {
                $title = ""
            }
            $heading = "## Slide $($slide.SlideIndex)"
            if (-not [string]::IsNullOrWhiteSpace($title)) {
                $heading += ": $($title.Replace("`n", " "))"
            }
            $result.Add($heading)
            $result.Add("")
            $result.Add("<!-- slide: $($slide.SlideIndex) -->")
            $result.Add("")

            $blocks = New-Object System.Collections.Generic.List[object]
            foreach ($shape in $slide.Shapes) {
                Add-PowerPointShapeContent $shape $blocks $titleShapeId
            }
            foreach ($block in $blocks | Sort-Object Top, Left) {
                foreach ($line in $block.Lines) {
                    $result.Add($line)
                }
                $result.Add("")
            }
        }
        return $result.ToArray()
    } finally {
        if ($null -ne $presentation) {
            $presentation.Close()
        }
    }
}

function Resolve-SemanticOutputPath {
    param(
        [string]$VirtualPath,
        [string]$KnowledgeRoot
    )

    $normalizedPath = $VirtualPath.Trim().Replace("\", "/")
    if (-not $normalizedPath.StartsWith("/")) {
        throw "semantic path must start with '/': $VirtualPath"
    }
    if (-not $normalizedPath.EndsWith(".md", [StringComparison]::OrdinalIgnoreCase)) {
        throw "semantic path must end with '.md': $VirtualPath"
    }

    $segments = @($normalizedPath.TrimStart("/").Split("/"))
    $invalidFileNameCharacters = [IO.Path]::GetInvalidFileNameChars()
    foreach ($segment in $segments) {
        if (
            [string]::IsNullOrWhiteSpace($segment) -or
            $segment -in @(".", "..") -or
            $segment.IndexOfAny($invalidFileNameCharacters) -ge 0
        ) {
            throw "semantic path contains an invalid segment: $VirtualPath"
        }
    }

    $resolvedRoot = [IO.Path]::GetFullPath($KnowledgeRoot).TrimEnd("\", "/")
    $relativePath = $normalizedPath.TrimStart("/").Replace(
        "/",
        [string][IO.Path]::DirectorySeparatorChar
    )
    $resolvedPath = [IO.Path]::GetFullPath((Join-Path $resolvedRoot $relativePath))
    $rootPrefix = $resolvedRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "semantic path resolves outside the knowledge root: $VirtualPath"
    }
    return $resolvedPath
}

if (-not (Test-Path -LiteralPath $sourceManifestPath)) {
    throw "source manifest does not exist: $sourceManifestPath"
}
if (-not (Test-Path -LiteralPath $SemanticPathMappingPath)) {
    throw "semantic path mapping does not exist: $SemanticPathMappingPath"
}

$semanticMapping = Get-Content -Raw -LiteralPath $SemanticPathMappingPath -Encoding UTF8 |
    ConvertFrom-Json
$semanticMappingRows = @($semanticMapping.items)
if ($semanticMappingRows.Count -ne 16) {
    throw "expected 16 semantic path mappings, got $($semanticMappingRows.Count)"
}

$semanticMappingBySampleId = @{}
$sampleIdBySemanticPath = @{}
foreach ($mappingRow in $semanticMappingRows) {
    $mappingSampleId = [string]$mappingRow.sample_id
    $semanticPath = [string]$mappingRow.semantic_path
    $contentLocatorType = [string]$mappingRow.content_locator_type
    if ([string]::IsNullOrWhiteSpace($mappingSampleId)) {
        throw "semantic path mapping contains an empty sample ID"
    }
    if ([string]::IsNullOrWhiteSpace($contentLocatorType)) {
        throw "semantic path mapping contains an empty locator type: $mappingSampleId"
    }
    if ($semanticMappingBySampleId.ContainsKey($mappingSampleId)) {
        throw "duplicate sample ID in semantic path mapping: $mappingSampleId"
    }

    Resolve-SemanticOutputPath $semanticPath $outputRoot | Out-Null
    $semanticPathKey = $semanticPath.ToLowerInvariant()
    if ($sampleIdBySemanticPath.ContainsKey($semanticPathKey)) {
        throw "duplicate semantic path in mapping: $semanticPath"
    }
    $semanticMappingBySampleId[$mappingSampleId] = $mappingRow
    $sampleIdBySemanticPath[$semanticPathKey] = $mappingSampleId
}

$documentTypes = @("excel", "html", "md", "pdf", "ppt", "word")
$allSourceRows = @(
    Get-Content -LiteralPath $sourceManifestPath -Encoding UTF8 |
        Where-Object { $_.Trim() } |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $documentTypes -contains $_.resource_type } |
        Sort-Object resource_type, sample_id
)
if ($allSourceRows.Count -ne 16) {
    throw "expected 16 document sources, got $($allSourceRows.Count)"
}
foreach ($sourceRow in $allSourceRows) {
    if (-not $semanticMappingBySampleId.ContainsKey([string]$sourceRow.sample_id)) {
        throw "semantic path mapping is missing sample ID: $($sourceRow.sample_id)"
    }
}
foreach ($mappingSampleId in $semanticMappingBySampleId.Keys) {
    if ($allSourceRows.sample_id -notcontains $mappingSampleId) {
        throw "semantic path mapping contains an unknown sample ID: $mappingSampleId"
    }
}
$sourceRows = @($allSourceRows)
if ($SampleId.Count -gt 0) {
    $sourceRows = @($allSourceRows | Where-Object { $SampleId -contains $_.sample_id })
    if ($sourceRows.Count -ne $SampleId.Count) {
        throw "one or more requested sample IDs were not found"
    }
}

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$word = $null
$excel = $null
$powerPoint = $null
$conversionRows = New-Object System.Collections.Generic.List[object]
$selectedIds = @($sourceRows.sample_id)
if (Test-Path -LiteralPath $conversionManifestPath) {
    Get-Content -LiteralPath $conversionManifestPath -Encoding UTF8 |
        Where-Object { $_.Trim() } |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $selectedIds -notcontains $_.sample_id } |
        ForEach-Object { $conversionRows.Add($_) }
}
$errors = New-Object System.Collections.Generic.List[string]

try {
    foreach ($sourceRow in $sourceRows) {
        $sourcePath = Join-Path $datasetRoot ([string]$sourceRow.local_path)
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            throw "source file does not exist: $sourcePath"
        }
        $mappingRow = $semanticMappingBySampleId[[string]$sourceRow.sample_id]
        $semanticPath = [string]$mappingRow.semantic_path
        $contentLocatorType = [string]$mappingRow.content_locator_type
        $extension = [IO.Path]::GetExtension($sourcePath).TrimStart(".").ToLowerInvariant()
        if ($extension -ne ([string]$mappingRow.file_format).ToLowerInvariant()) {
            throw "mapped file format does not match source: $($sourceRow.sample_id)"
        }
        $startedAt = Get-Date
        try {
            switch ($sourceRow.resource_type) {
                "md" {
                    $bodyText = [IO.File]::ReadAllText($sourcePath, [Text.Encoding]::UTF8)
                    $bodyLines = @($bodyText.Replace("`r`n", "`n").Replace("`r", "`n").Split("`n"))
                    $parser = "direct_text"
                }
                "excel" {
                    if ($null -eq $excel) {
                        $excel = New-Object -ComObject Excel.Application
                        $excel.Visible = $false
                        $excel.DisplayAlerts = $false
                        $excel.AskToUpdateLinks = $false
                    }
                    $bodyLines = @(Convert-ExcelWorkbook $excel $sourcePath)
                    $parser = "excel_com"
                }
                "ppt" {
                    if ($null -eq $powerPoint) {
                        $powerPoint = New-Object -ComObject PowerPoint.Application
                    }
                    $bodyLines = @(Convert-PowerPointPresentation $powerPoint $sourcePath)
                    $parser = "powerpoint_com"
                }
                default {
                    if ($null -eq $word) {
                        $word = New-Object -ComObject Word.Application
                        $word.Visible = $false
                        $word.DisplayAlerts = 0
                        $word.Options.UpdateLinksAtOpen = $false
                    }
                    $includePageMarkers = $sourceRow.resource_type -in @("pdf", "word")
                    $bodyLines = @(Convert-WordLikeDocument $word $sourcePath $includePageMarkers)
                    $parser = "word_com"
                }
            }

            $title = [string]$sourceRow.title
            if ([string]::IsNullOrWhiteSpace($title)) {
                $title = [IO.Path]::GetFileNameWithoutExtension($sourcePath)
            }
            $markdown = New-SourceMarkdown `
                -Title $title `
                -FileFormat $extension `
                -SampleId ([string]$sourceRow.sample_id) `
                -ContentLocatorType $contentLocatorType `
                -BodyLines $bodyLines
            $outputPath = Resolve-SemanticOutputPath $semanticPath $outputRoot
            $outputDirectory = Split-Path -Parent $outputPath
            New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
            [IO.File]::WriteAllText($outputPath, $markdown, $utf8NoBom)

            $legacyTypeRoot = Join-Path $outputRoot ([string]$sourceRow.resource_type)
            $legacyOutputPath = Join-Path $legacyTypeRoot ("$($sourceRow.sample_id).md")
            if (
                $legacyOutputPath -ne $outputPath -and
                (Test-Path -LiteralPath $legacyOutputPath -PathType Leaf)
            ) {
                Remove-Item -LiteralPath $legacyOutputPath -Force
            }
            $duration = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 3)
            $conversionRows.Add([ordered]@{
                schema = "yunzhi-source-markdown-conversion-v2"
                sample_id = $sourceRow.sample_id
                resource_type = $sourceRow.resource_type
                file_format = $extension
                content_locator_type = $contentLocatorType
                semantic_path = $semanticPath
                source_path = $sourcePath.Substring($ProjectRoot.Length + 1).Replace("\", "/")
                source_sha256 = $sourceRow.sha256
                output_path = $outputPath.Substring($ProjectRoot.Length + 1).Replace("\", "/")
                output_sha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
                output_characters = $markdown.Length
                output_lines = @($markdown.Split("`n")).Count
                parser = $parser
                duration_seconds = $duration
                status = "success"
            })
            Write-Output "converted $($sourceRow.resource_type) $($sourceRow.sample_id)"
        } catch {
            $message = "$($sourceRow.sample_id): $($_.Exception.Message)"
            $errors.Add($message)
            $conversionRows.Add([ordered]@{
                schema = "yunzhi-source-markdown-conversion-v2"
                sample_id = $sourceRow.sample_id
                resource_type = $sourceRow.resource_type
                file_format = $extension
                content_locator_type = $contentLocatorType
                semantic_path = $semanticPath
                source_path = $sourcePath.Substring($ProjectRoot.Length + 1).Replace("\", "/")
                status = "error"
                error = $_.Exception.Message
            })
            Write-Warning $message
        }
    }
} finally {
    if ($null -ne $word) { $word.Quit() }
    if ($null -ne $excel) { $excel.Quit() }
    if ($null -ne $powerPoint) { $powerPoint.Quit() }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$manifestLines = @($conversionRows | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 6 })
[IO.File]::WriteAllLines($conversionManifestPath, $manifestLines, $utf8NoBom)

if ($errors.Count -gt 0) {
    throw "document conversion failed for $($errors.Count) source(s)"
}

foreach ($documentType in $documentTypes) {
    $legacyTypeRoot = Join-Path $outputRoot $documentType
    if (
        (Test-Path -LiteralPath $legacyTypeRoot -PathType Container) -and
        @(Get-ChildItem -LiteralPath $legacyTypeRoot -Force).Count -eq 0
    ) {
        Remove-Item -LiteralPath $legacyTypeRoot -Force
    }
}

Write-Output "conversion manifest: $conversionManifestPath"

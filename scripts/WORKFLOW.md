# convert_multimodal.py Workflow

## End-to-End Pipeline

```
signal/                                           scripts/
  p1_methodical_T-01/                              convert_multimodal.py
  p1_methodical_T-02/                                     |
  ...                                                     |
  p20_visual_auditor_T-20/            <--- discovers 640 trajectories
                                                          |
                                                          v
                                              +-----------------------+
                                              |   For each trajectory |
                                              |     (sequential)      |
                                              +-----------+-----------+
                                                          |
                                                          v
                                              +-----------------------+
                                              | load_content_blobs()  |
                                              | read manifest.json    |
                                              | filter fs_snapshot    |
                                              | load .blob text       |
                                              +-----------+-----------+
                                                          |
                                                          v
                                              +-----------------------+
                                              | ThreadPoolExecutor    |
                                              | (parallel per blob)   |
                                              +-----------+-----------+
                                                          |
                                          +---------------+---------------+
                                          |                               |
                                          v                               v
                                   blob #1 (hash_a)                blob #2 (hash_b)
                                          |                               |
                                          v                               v
                                  convert_single_blob()           convert_single_blob()
                                          |                               |
                                          +-------------------------------+
                                                          |
                                                          v
                                              +-----------------------+
                                              |   Write manifest.json |
                                              |   (incremental merge) |
                                              +-----------------------+
```

## Single Blob Conversion Detail

```
                              +------------------+
                              |   .blob file     |
                              | (MD / TXT / CSV) |
                              +--------+---------+
                                       |
                      +----------------+----------------+
                      |                                 |
                      v                                 v
            +------------------+              +------------------+
            |  generate_html() |              |  analyze_blob()  |
            |  markdown lib    |              |  Gemini 2.5 Flash|
            |  (no LLM needed) |              |  structured JSON |
            +--------+---------+              +--------+---------+
                     |                                 |
                     v                                 v
               +----------+                  +-----------------+
               | .html    |                  | Structured JSON |
               +----------+                  +--------+--------+
                                                      |
              Gemini decides "modalities" field        |
              e.g. ["html","pdf","docx","mp3","png"]   |
                                                       |
            +------+------+------+------+------+-------+
            |      |      |      |      |      |
            v      v      v      v      v      v
         +-----+ +----+ +----+ +----+ +----+ +----+
         | PDF | |DOCX| |XLSX| |PPTX| |MP3 | |PNG |
         +-----+ +----+ +----+ +----+ +----+ +----+
```

## Generator Details

```
+------------------------------------------------------------------+
|                    Structured JSON from Gemini                    |
|                                                                  |
|  {                                                               |
|    "title":        "..."         ----+---> PDF title, DOCX h0    |
|    "language":     "zh|en"       ----|---> TTS lang, font choice |
|    "summary":      "2-3 lines"  ----|---> MP3 (gTTS)            |
|    "text_content": "clean text"  ----|---> PDF/DOCX body         |
|    "sections":     [{h,c},...]   ----|---> PPTX slides, DOCX h1  |
|    "table_headers/rows":         ----|---> XLSX                  |
|    "chart_type/labels/values":   ----|---> PNG (matplotlib)      |
|    "key_quote":    "..."         ----|---> PNG (Pillow quote card)|
|    "modalities":   [...]         ----+---> which generators run  |
|  }                                                               |
+------------------------------------------------------------------+

  Generator        Library          Input Field          Output
  ----------------------------------------------------------------
  generate_html    markdown         blob.text (raw)      {hash}.html
  generate_pdf     fpdf2            text_content         {hash}.pdf
  generate_docx    python-docx      sections             {hash}.docx
  generate_xlsx    openpyxl         table_headers/rows   {hash}.xlsx
  generate_pptx    python-pptx      sections             {hash}.pptx
  generate_mp3     gTTS             summary              {hash}.mp3
  generate_png     matplotlib       chart_data           {hash}.png
                   Pillow           key_quote            {hash}.png
```

## File I/O Map

```
INPUT (per trajectory)
======================

signal/{profile}_{task}/
  +-- media/
       +-- manifest.json          <--- read: discover blob hashes
       +-- blobs/
            +-- {hash_a}.blob     <--- read: raw text content
            +-- {hash_b}.blob
            +-- ...


OUTPUT (per trajectory)
=======================

signal/{profile}_{task}/
  +-- media/
       +-- blobs_multimodal/      <--- NEW directory
            +-- manifest.json     <--- generated: conversion metadata
            +-- {hash_a}.html
            +-- {hash_a}.pdf
            +-- {hash_a}.docx
            +-- {hash_a}.xlsx     (only if has_table)
            +-- {hash_a}.pptx     (only if 2+ sections)
            +-- {hash_a}.mp3      (only if summary >= 20 chars)
            +-- {hash_a}.png      (chart or quote card)
            +-- {hash_b}.html
            +-- {hash_b}.pdf
            +-- ...
```

## Error Handling Flow

```
convert_single_blob(blob)
         |
         +---> generate_html()  -----> always runs, no LLM
         |          |
         |          +-- fail? --> record error, continue
         |
         +---> analyze_blob()   -----> Gemini API call
         |          |
         |          +-- fail? --> return (HTML only)
         |          +-- ok?   --> proceed with generators
         |
         +---> for each modality in Gemini's "modalities":
                    |
                    +---> generator()
                    |         |
                    |         +-- fail? --> record error, continue next
                    |         +-- ok?   --> record in result.modalities
                    |
                    +---> (next modality)

  * Each blob is independent -- one failure does not block others
  * Each generator is independent -- one failure does not block others
  * All errors recorded in manifest.json per-entry "errors" field
```

## CJK Font Resolution

```
  Is text Chinese?
       |
       +-- No ----> Songti available? --+-- Yes -> use Songti (handles em-dash etc.)
       |                                +-- No  -> Helvetica (strip non-latin1)
       |
       +-- Yes ---> Songti available? --+-- Yes -> use Songti
                                        +-- No  -> SKIP pdf/png generation

  Font search order:
    1. /System/Library/Fonts/Supplemental/Songti.ttc
    2. /System/Library/Fonts/STHeiti Medium.ttc
    3. /System/Library/Fonts/STHeiti Light.ttc
    4. /Library/Fonts/Arial Unicode.ttf
```

## CLI Options

```
python scripts/convert_multimodal.py [OPTIONS]

  --parallel N, -p N       Workers per trajectory (default: 4)
                           Blobs within one trajectory run in parallel.
                           Trajectories themselves run sequentially
                           (to respect Gemini rate limits).

  --trajectory NAME, -t    Process single trajectory only
                           e.g. -t p4_structured_analyst_T-20

  --skip-existing, -s      Skip blobs already in blobs_multimodal/manifest.json
                           Safe for incremental / resume runs.

  --dry                    Preview mode: list blobs per trajectory, no conversion.
```

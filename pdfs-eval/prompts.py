# DEFAULT_PROMPT = """Extract the text from the document in Markdown format, and extract the tables in HTML format. 
# Do not add style or anything, just the text. Do not ever generate tables in markdown format. Give me the output, nothing else."""


DEFAULT_PROMPT = """Extract all text and tabular data from this document image with the following requirements:

TEXT EXTRACTION:
- Convert all text content to clean Markdown format
- Preserve document structure using appropriate headers (# ## ###)
- Maintain paragraph breaks and line spacing
- Keep bullet points and numbered lists in proper Markdown syntax
- Preserve text formatting (bold, italic) where clearly visible
- Include all readable text including headers, footers, captions, and annotations

TABLE EXTRACTION:
- Extract ALL tables as clean HTML format using <table>, <tr>, <td>, <th> tags only
- Do NOT use any CSS styles, classes, or attributes
- Do NOT convert tables to Markdown format under any circumstances
- Ensure proper table structure with headers in <th> tags
- Merge cells should use colspan/rowspan attributes if needed
- Preserve numerical data and alignment

OUTPUT FORMAT:
- Return ONLY the extracted content
- No explanatory text, comments, or metadata
- No code block markers (```)
- Tables should be standalone HTML elements
- Text content in Markdown format

QUALITY REQUIREMENTS:
- Maintain original document meaning and context
- Ensure all visible text is captured
- Handle multi-column layouts appropriately
- Process mathematical formulas and special characters accurately
- Ignore watermarks, page numbers unless they're part of main content"""
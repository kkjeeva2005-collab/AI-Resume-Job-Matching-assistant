import fitz


def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    return extract_text_from_pdf_bytes(pdf_bytes)


def extract_text_from_pdf_bytes(pdf_bytes):
    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in document:
        page_text = page.get_text()

        if page_text:
            text += page_text + "\n"

    document.close()

    return text


def clean_text(text):

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
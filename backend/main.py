import ollama
import pypdf
import pandas

def extract_pdf(path) -> str:
    # TODO handle I/O failures
    reader = pypdf.PdfReader(path)
    output = ""

    for page in reader.pages:
        output += page.extract_text(
            extraction_mode="layout",
            layout_mode_scale_weight=1.2
        )

    return output

def extract_excel(path) -> str:
    # TODO handle I/O failures
    return pandas.read_excel(path, index_col=None).to_csv(encoding="utf-8")

def extract_speech(path) -> str:
    # TODO
    pass

def summarize(text):
    # TODO handle I/O failures
    stream = ollama.chat(
        model='phi3.5',
        messages=[{'role': 'user', 'content': "Summarize the following document:\n===\n%s\n===" % (text)}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)



def main():
    print("Hello from backend!")


if __name__ == "__main__":
    main()
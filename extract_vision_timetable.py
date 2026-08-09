import os
import base64
import requests
import fitz  # PyMuPDF

def get_openrouter_api_key(directory):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        env_path = os.path.join(directory, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        key = line.strip().split("=", 1)[1]
    return key

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(directory, "7th semester Time Table 2026.pdf")
    output_path = os.path.join(directory, "timetable_2026.txt")
    
    api_key = get_openrouter_api_key(directory)
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set in .env")
        return

    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return

    print(f"Opening {pdf_path} with PyMuPDF...")
    doc = fitz.open(pdf_path)
    
    extracted_texts = []
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    url = "https://openrouter.ai/api/v1/chat/completions"

    for i in range(len(doc)):
        print(f"Processing page {i + 1} of {len(doc)}...")
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # high resolution
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        
        prompt = (
            "This is a scanned image of a college timetable. Please transcribe the timetable accurately "
            "into Markdown tables. Make sure to capture all details such as Section, Day, Time, "
            "Subject, and Faculty names. Maintain the structure so it is extremely clear which class "
            "belongs to which section and which time."
        )

        data = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        print("Sending image to OpenRouter Vision API (google/gemini-1.5-pro)...")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result_text = response.json()["choices"][0]["message"]["content"]
            extracted_texts.append(f"### Page {i + 1} Timetable\n\n{result_text}")
            print(f"Successfully transcribed page {i + 1}.")
        else:
            print(f"Error from API on page {i + 1}: {response.status_code} - {response.text}")

    if extracted_texts:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(extracted_texts))
        print(f"\nAll transcriptions saved to: {output_path}")
        print("You can now run 'python update_index.py' to embed this text!")
    else:
        print("No text was extracted.")

if __name__ == "__main__":
    main()

import os
import pdfplumber
import json

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(directory, "7th semester Time Table 2026.pdf")
    output_path = os.path.join(directory, "timetable_dump.txt")
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return
        
    print(f"Reading {pdf_path}...")
    
    with pdfplumber.open(pdf_path) as pdf:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(pdf.pages):
                f.write(f"--- PAGE {i + 1} ---\n")
                
                # Extract simple text to see how it looks
                text = page.extract_text()
                f.write("=== RAW TEXT ===\n")
                f.write(str(text) + "\n\n")
                
                # Extract tables
                tables = page.extract_tables()
                f.write(f"=== TABLES FOUND: {len(tables)} ===\n")
                for j, table in enumerate(tables):
                    f.write(f"  Table {j + 1}:\n")
                    # Write as JSON for easy reading of the structure
                    f.write(json.dumps(table, indent=2) + "\n\n")
                    
                f.write("\n\n")
                
    print(f"Dumped timetable contents to {output_path}")

if __name__ == "__main__":
    main()

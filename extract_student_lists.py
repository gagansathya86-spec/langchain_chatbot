import os
import csv
import pdfplumber
import re

def extract_students_from_pdf(pdf_path, subject, faculty, sections):
    students = []
    
    # Regex to catch USNs like 1DB23CS039
    usn_pattern = re.compile(r"1DB\d{2}[A-Z]{2}\d{3}", re.IGNORECASE)
    
    print(f"Parsing {os.path.basename(pdf_path)}...")
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    # Clean row from None types
                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                    
                    # Find USN and Name in the row
                    usn = None
                    name = None
                    usn_index = -1
                    
                    # Look for USN anywhere in the row
                    for i, cell in enumerate(clean_row):
                        if usn_pattern.search(cell):
                            usn = usn_pattern.search(cell).group()
                            usn_index = i
                            break
                    
                    if usn:
                        # The student name is usually in the column immediately after the USN
                        if usn_index + 1 < len(clean_row) and len(clean_row[usn_index + 1]) > 2:
                            name = clean_row[usn_index + 1]
                        else:
                            # Fallback if the next column is empty
                            potential_names = [cell for cell in clean_row[usn_index+1:] if len(cell) > 3 and not "Analytics" in cell and not "Learning" in cell]
                            if potential_names:
                                name = potential_names[0]
                                
                        if name:
                            name = name.replace('\n', ' ').strip()
                            
                            students.append({
                                "USN": usn,
                                "Name": name,
                                "Subject": subject,
                                "Faculty": faculty,
                                "Sections": sections
                            })
                            
    return students

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(directory, "Students_2026_V2.csv")
    
    # Define our PDFs and their metadata
    files_to_process = [
        {
            "filename": "BDA-Dr. Usha Kirana-A&B.pdf",
            "subject": "Big Data Analytics",
            "faculty": "Dr. Usha Kirana",
            "sections": "A & B"
        },
        {
            "filename": "BDA-Ms. Sinchana K P-B&C.pdf",
            "subject": "Big Data Analytics",
            "faculty": "Ms. Sinchana K P",
            "sections": "B & C"
        },
        {
            "filename": "DL-Dr. Md. Najmus Saqhib-A,B,C&D.pdf",
            "subject": "Deep Learning",
            "faculty": "Dr. Md. Najmus Saqhib",
            "sections": "A, B, C & D"
        }
    ]
    
    all_students = []
    
    for item in files_to_process:
        pdf_path = os.path.join(directory, item["filename"])
        if os.path.exists(pdf_path):
            extracted = extract_students_from_pdf(
                pdf_path, 
                item["subject"], 
                item["faculty"], 
                item["sections"]
            )
            all_students.extend(extracted)
            print(f"Extracted {len(extracted)} students from {item['filename']}")
        else:
            print(f"Warning: Could not find {pdf_path}")
            
    if all_students:
        print(f"Writing {len(all_students)} total records to {output_csv}...")
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["USN", "Name", "Subject", "Faculty", "Sections"])
            writer.writeheader()
            for student in all_students:
                writer.writerow(student)
        print("Done! You can now update your vector index.")
    else:
        print("No students were extracted.")

if __name__ == "__main__":
    main()

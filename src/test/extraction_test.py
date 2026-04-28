from src.extraction import extraire_cv, extraire_offre
import json

# Test DOCX
#print("=== Test DOCX ===")
#result_docx = extraire_cv("cv_test.docx")
#print(json.dumps(result_docx, indent=2, ensure_ascii=False))

# Test PDF
print("\n=== Test PDF ===")
result_pdf = extraire_cv("2.pdf")
print(json.dumps(result_pdf, indent=2, ensure_ascii=False))

# Test Offre (JSON simulé)
#print("\n=== Test Offre ===")
#json_offre = json.dumps({"offre": "Ingénieur Python, 3 ans d'expérience, basé à Oujda"})
#result_offre = extraire_offre(json_offre)
#print(json.dumps(result_offre, indent=2, ensure_ascii=False))

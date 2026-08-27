from app.services.parser import Parser

ocr_results = [
    {'class': 'pan',
    'text': 'OTMPK6463G',
    'confidence': 0.9997822642326355},

    {'class': 'name',
    'text': 'ADITYA KUMAR',
    'confidence': 0.999258816242218}, 

    {'class': 'dob',
    'text': '14/12/2005', 
    'confidence': 0.9999966621398926},

    {'class': 'father', 
    'text': 'ANIL KUMAR', 
    'confidence': 0.9999755024909973}
]

parser = Parser()

result = parser.parse(ocr_results)

print(result)
import requests
import pytesseract
from PIL import Image
import io
import fitz  # PyMuPDF
import tempfile
import os
from typing import Dict, List, Union

def extract_text_from_image(image_url: str) -> str:
    """Extract text from image using OCR"""
    try:
        # Download image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        
        # Perform OCR
        text = pytesseract.image_to_string(image, lang='vie')  # 'vie' for Vietnamese
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_url: str) -> str:
    """Extract text from PDF"""
    try:
        # Download PDF
        response = requests.get(pdf_url)
        
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Extract text from PDF
        text = ""
        with fitz.open(temp_path) as pdf_doc:
            for page in pdf_doc:
                text += page.get_text()
        
        # Clean up temporary file
        os.unlink(temp_path)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def calculate_relevance_score(query: str, result: Dict) -> float:
    """Calculate relevance score for a search result"""
    query_terms = set(query.lower().split())
    
    # Get text from different fields
    title = result.get('title', '').lower()
    snippet = result.get('snippet', '').lower()
    url = result.get('link', '').lower()
    
    # Calculate exact phrase matches (for multi-word terms)
    query_lower = query.lower()
    exact_phrase_score = 0
    if query_lower in title:
        exact_phrase_score += 4
    if query_lower in snippet:
        exact_phrase_score += 2
    
    # Calculate individual word matches
    title_matches = sum(1 for term in query_terms if term in title)
    snippet_matches = sum(1 for term in query_terms if term in snippet)
    
    # Higher score for educational/admission related pages
    context_score = 0
    edu_indicators = {'điểm chuẩn', 'tuyển sinh', 'ngành', 'đại học', 'ts.hust', 'bach khoa', 'TSA','đánh giá tư duy','kỳ thi tư duy', 'xét tuyển tài năng'}
    for indicator in edu_indicators:
        if indicator in url:
            context_score += 1
        if indicator in title:
            context_score += 1
            
    total_score = (title_matches * 2) + snippet_matches + exact_phrase_score + context_score
    return total_score

def google_search(query: str, api_key: str, search_engine_id: str, num_results: int = 1):
    """Enhanced Google search with image and PDF processing"""
    url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query,
        'num': num_results,
    }
    
    headers = {
        'Accept': 'application/json',
        'Referer': 'https://www.google.com'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if 'items' in data:
                results = []
                for item in data['items']:
                    result = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    }
                    
                    # Process file based on type
                    file_url = item.get('link', '')
                    if file_url.lower().endswith(('.png', '.jpg', '.jpeg')):
                        result['extracted_text'] = extract_text_from_image(file_url)
                    elif file_url.lower().endswith('.pdf'):
                        result['extracted_text'] = extract_text_from_pdf(file_url)
                    
                    results.append(result)
                
                # Sort results by relevance score
                results.sort(key=lambda x: calculate_relevance_score(query, x), reverse=True)
                return results[:1] if results else []
            return []
        return []
            
    except Exception as e:
        return []

def test_search(question):
    API_KEY = "AIzaSyCqt0PNREnSBj5i4tPck91LIsoZJfGxnU0"
    SEARCH_ENGINE_ID = "d3776215e80c949ef"
    
    test_query = question
    results = google_search(test_query, API_KEY, SEARCH_ENGINE_ID)
    result = results[0]
    return result['link']

if __name__ == "__main__":
    question = "Em muốn tìm hiểu về chương trình Kỹ thuật Sinh học, em có thể liên hệ với ai ạ?"
    print(test_search(question))
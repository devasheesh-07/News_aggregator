import collections
import collections.abc
collections.Sequence = collections.abc.Sequence

from flask import Flask, render_template, request
from sumy.parsers.plaintext import PlaintextParser
# ... other imports


from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import nltk 
from dotenv import load_dotenv
import os 

load_dotenv()

app = Flask(__name__)

# Initialize NLTK data
def initialize_nltk():
    try:
        # Set custom download path
        nltk_dir = os.path.join(os.path.expanduser('~'), 'nltk_data_custom')
        os.makedirs(nltk_dir, exist_ok=True)
        nltk.data.path.append(nltk_dir)
        
        # Download specific required resources
        resources = ['punkt', 'punkt_tab', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, download_dir=nltk_dir)
    except Exception as e:
        print(f"NLTK initialization error: {e}")

# Call this before your Flask app runs
initialize_nltk()
# Run initialization
#initialize_nltk()

# MongoDB setup
# Connect to MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['tech_aggregator']
summaries_collection = db['summaries']

# Load sentence transformer model for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_valid_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_youtube_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\/]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^\/]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^\/\?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/live\/([^\/\?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id):
    """Get transcript for YouTube video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None

def extract_article_text(url):
    """Extract main text content from article/blog"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.decompose()
            
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text if text else None
    except Exception as e:
        print(f"Error extracting article text: {e}")
        return None

def summarize_text(text, sentences_count=5):
    """Summarize text using LSA algorithm"""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return ' '.join([str(sentence) for sentence in summary])
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def store_summary(url, summary):
    """Store summary in MongoDB with vector embedding"""
    if not url or not summary:
        return False
    
    try:
        summary_doc = {
            'url': url,
            'summary': summary,
            'embedding': model.encode(summary).tolist()
        }
        summaries_collection.insert_one(summary_doc)
        return True
    except Exception as e:
        print(f"Error storing summary: {e}")
        return False

def search_summaries(query):
    """Semantic search through stored summaries"""
    if not query:
        return []
    
    try:
        query_embedding = model.encode(query).reshape(1, -1)
        summaries = list(summaries_collection.find({}, {'_id': 0}))
        
        if not summaries:
            return []
        
        embeddings = np.array([summary['embedding'] for summary in summaries])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        results = sorted(zip(summaries, similarities), key=lambda x: x[1], reverse=True)
        return [result[0] for result in results[:5]]
    except Exception as e:
        print(f"Error searching summaries: {e}")
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        
        if not url:
            return render_template('index.html', error="Please enter a URL")
        
        if not is_valid_url(url):
            return render_template('index.html', error="Invalid URL format")
        
        try:
            # YouTube processing
            if 'youtube.com' in url or 'youtu.be' in url:
                video_id = extract_youtube_video_id(url)
                if not video_id:
                    return render_template('index.html', error="Invalid YouTube URL")
                
                transcript = get_youtube_transcript(video_id)
                if not transcript:
                    return render_template('index.html', error="Could not get transcript for this video")
                
                summary = summarize_text(transcript)
                if not summary:
                    return render_template('index.html', error="Could not generate summary")
                
                if store_summary(url, summary):
                    return redirect(url_for('results', summary=summary, url=url))
                else:
                    return render_template('index.html', error="Failed to store summary")
            
            # Blog/article processing
            article_text = extract_article_text(url)
            if not article_text:
                return render_template('index.html', error="Could not extract article text")
            
            summary = summarize_text(article_text)
            if not summary:
                return render_template('index.html', error="Could not generate summary")
            
            if store_summary(url, summary):
                return redirect(url_for('results', summary=summary, url=url))
            else:
                return render_template('index.html', error="Failed to store summary")
        
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    
    return render_template('index.html')

@app.route('/results')
def results():
    summary = request.args.get('summary', '')
    url = request.args.get('url', '')
    return render_template('results.html', summary=summary, url=url)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').strip()
    if not query:
        return render_template('results.html', search_error="Please enter a search query")
    
    results = search_summaries(query)
    return render_template('results.html', search_results=results, query=query) 

@app.route('/history')
def history():
    """Show all previously summarized URLs"""
    try:
        summaries = list(summaries_collection.find({}, {'_id': 0, 'url': 1, 'summary': 1}).sort('_id', -1))
        return render_template('history.html', summaries=summaries)
    except Exception as e:
        return render_template('history.html', error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
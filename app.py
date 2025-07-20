from flask import Flask, render_template, request, jsonify, flash
import os
from werkzeug.utils import secure_filename
import traceback
from typing import List, Dict, Optional
from SearchEngine.utils._logger import get_logger
from SearchEngine.utils._data_types import SearchMode, SearchResult, EmbeddingType
from SearchEngine.main import SearchEngine


logger = get_logger()

app = Flask(__name__)
app.secret_key = os.urandom(24) 

UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 

search_engine: Optional[SearchEngine] = None
openai_api_key = os.getenv("OPENAI_API_KEY", None) 

def initialize_search_engine():
    global search_engine
    if search_engine is None:
        try:
            logger.info("Initializing Search Engine...")
            search_engine = SearchEngine(
                path=UPLOAD_FOLDER, 
                model_type=EmbeddingType.MINILM_L6, 
                openai_apikey=openai_api_key,
                temp=False 
            )
            logger.info("Search engine initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Fatal error initializing search engine: {str(e)}", exc_info=True)
            return False


@app.before_request
def ensure_engine_initialized():
    if search_engine is None:
        if not initialize_search_engine():
             pass 


def get_search_engine() -> Optional[SearchEngine]:
    """Safely gets the search engine instance, handling initialization errors."""
    if search_engine is None:
        flash("Search engine is not available. Initialization may have failed. Check server logs.", "error")
        return None
    return search_engine

def format_results_for_json(results: List[SearchResult]) -> List[Dict]:
    """Converts SearchResult objects to JSON-serializable dictionaries."""
    output = []
    for r in results:
        output.append({
            "doc_id": r.doc_id,
            "file_name": r.file_name,
            "bm25_score": float(r.bm25_score) if r.bm25_score is not None else None,
            "semantic_score": float(r.semantic_score) if r.semantic_score is not None else None,
            "combined_score": float(r.combined_score) if r.combined_score is not None else None,
            "positions": r.positions,
            "all_terms_present": r.all_terms_present,
            "chunk_id": [int(x) for x in r.chunk_id] if r.chunk_id is not None else [],
            "chunk_text": r.chunk_text,
        })
    return output

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', query="")



@app.route('/api/search', methods=['POST'])
def api_search():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    try:
        query = request.form.get('query', '')
        mode_str = request.form.get('mode', 'hybrid')
        k = int(request.form.get('k', 5))

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        mode_map = {
            'hybrid':  SearchMode.HYBRID,
            'semantic':SearchMode.SEMANTIC,
            'keyword': SearchMode.KEYWORD
        } 
        mode = mode_map.get(mode_str, SearchMode.HYBRID)

        results = engine.search(query, k=k, mode=mode)
        formatted_results = format_results_for_json(results)
        return jsonify({"results": formatted_results})

    except ValueError as ve:
         logger.warning(f"Search input error: {ve}")
         return jsonify({"error": f"Invalid input: {ve}"}), 400
    except Exception as e:
        logger.error(f"Search API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"An internal error occurred during search."}), 500


@app.route('/api/qa', methods=['POST'])
def api_qa_search():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503
    if not engine.openai_client:
         return jsonify({"error": "QA requires an OpenAI API key configured on the server."}), 400

    try:
        query = request.form.get('query', '')
        model = request.form.get('model', 'gpt-4o-mini')
        k = int(request.form.get('k', 3))

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        logger.info(f"API QA: query='{query}', model='{model}', k={k}")
        answer = engine.llm_qa_search(query, model=model, k=k)
        return jsonify({'answer': answer})

    except ValueError as ve:
         logger.warning(f"QA input error: {ve}")
         return jsonify({"error": f"Invalid input: {ve}"}), 400
    except Exception as e:
        logger.error(f"QA API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An internal error occurred during QA processing.'}), 500


@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('file')
    if not files or files[0].filename == '':
         return jsonify({'error': 'No selected file(s)'}), 400

    processed_files = []
    errors = []

    for file in files:
        if file: 
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                logger.info(f"File saved: {filepath}")
                success = engine.add_source(filepath)
                if success:
                    processed_files.append(filename)
                    logger.info(f"Successfully indexed: {filename}")
                else:
                    errors.append(f"Failed to index {filename}")
                    logger.error(f"Indexing failed for {filename}")

            except Exception as e:
                logger.error(f"Error uploading/indexing file {filename}: {str(e)}\n{traceback.format_exc()}")
                errors.append(f"Error processing {filename}: {str(e)}")

    response = {"processed": processed_files}
    if errors:
        response["errors"] = errors
        return jsonify(response), 500 
    else:
        return jsonify(response), 200


@app.route('/api/add_text', methods=['POST'])
def api_add_text():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    text = request.form.get('text', '')
    if not text or text.isspace():
        return jsonify({'error': 'No text provided'}), 400

    try:
        logger.info(f"API Add Text: Adding text snippet (length: {len(text)})")
        success = engine.add_source(text)
        if success:
            return jsonify({'success': True, 'message': 'Text added and indexed successfully!'})
        else:
            return jsonify({'error': 'Failed to index text snippet.'}), 500
    except Exception as e:
        logger.error(f"Add Text API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An internal error occurred while adding text.'}), 500


@app.route('/api/add_url', methods=['POST'])
def api_add_url():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    url = request.form.get('url', '')
    if not url or not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Invalid or missing URL provided'}), 400

    try:
        logger.info(f"API Add URL: Adding URL {url}")
        success = engine.add_source(url)
        if success:
            return jsonify({'success': True, 'message': 'URL added and indexed successfully!'})
        else:
            return jsonify({'error': 'Failed to process or index the URL.'}), 500
    except ImportError:
         logger.error("URL processing dependencies (requests, beautifulsoup4) not installed.")
         return jsonify({'error': 'URL processing libraries not installed on server.'}), 501
    except Exception as e:
        logger.error(f"Add URL API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An internal error occurred while adding the URL.'}), 500


@app.route('/api/settings/weights', methods=['POST'])
def api_set_weights():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    try:
        semantic_weight = float(request.form.get('semantic_weight', 0.5))
        keyword_weight = float(request.form.get('keyword_weight', 0.5))

        logger.info(f"API Set Weights: Semantic={semantic_weight}, Keyword={keyword_weight}")
        engine.set_search_weights(semantic_weight, keyword_weight)
        return jsonify({'success': True, 'message': 'Search weights updated.'})
    except (ValueError, AssertionError) as e:
        logger.warning(f"Set Weights error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Set Weights API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An internal error occurred while setting weights.'}), 500
    
    
@app.route('/api/add_domain', methods=['POST'])
def api_add_domain():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    url = request.form.get('domain_url', '')
    save_to_disk = request.form.get('save_to_disk', 'true').lower() == 'true'
    max_pages = int(request.form.get('max_pages', 20))

    if not url or not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Invalid or missing domain URL provided'}), 400

    try:
        logger.info(f"API Add Domain: Crawling domain {url} (max_pages={max_pages}, save_to_disk={save_to_disk})")
        success = engine.data_storage.read_from_domain(url, save_to_disk=save_to_disk, max_pages=max_pages)
        if success:
            return jsonify({'success': True, 'message': f'Domain crawl and indexing for {url} completed!'})
        else:
            return jsonify({'error': 'Failed to crawl or index the domain.'}), 500
    except Exception as e:
        logger.error(f"Add Domain API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An internal error occurred while crawling the domain.'}), 500


@app.route('/api/sources', methods=['GET'])
def api_list_sources():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    try:
        sources = engine.list_sources()
        return jsonify({"sources": sources})
    except Exception as e:
        logger.error(f"List Sources API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred while listing sources."}), 500


@app.route('/api/source/<int:doc_id>/content', methods=['GET'])
def api_get_source_content(doc_id):
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    try:
        logger.info(f"API Get Content: Requesting content for doc_id {doc_id}")
        content = engine.get_source_content(doc_id)

        if content.startswith("[Error retrieving"):
             return jsonify({"error": content}), 404 

        return jsonify({"doc_id": doc_id, "content": content})

    except Exception as e:
        logger.error(f"Get Source Content API error (doc_id: {doc_id}): {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"An internal error occurred while retrieving content for document {doc_id}."}), 500


@app.route('/api/source/<int:doc_id>/delete', methods=['DELETE'])
def api_delete_source(doc_id):
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    try:
        logger.warning(f"API Delete Source: Requesting deletion for doc_id {doc_id}")
        success = engine.delete_source(doc_id)
        if success:
            return jsonify({'success': True, 'message': f'Document {doc_id} deleted successfully.'})
        else:
            return jsonify({'error': f'Failed to delete document {doc_id}. It might not exist or an error occurred.'}), 404 # Or 500
    except Exception as e:
        logger.error(f"Delete Source API error (doc_id: {doc_id}): {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"An internal error occurred while deleting document {doc_id}."}), 500
    

@app.route('/api/source/delete_all', methods=['DELETE'])
def api_delete_all_sources():
    engine = get_search_engine()
    if not engine:
        return jsonify({"error": "Search engine not available"}), 503

    try:
        logger.warning("API Delete All Sources: Requesting deletion of all sources")
        success = engine.clear_index()
        if success:
            return jsonify({'success': True, 'message': 'All documents deleted successfully.'})
        else:
            return jsonify({'error': 'Failed to delete all documents. An error occurred.'}), 500
    except Exception as e:
        logger.error(f"Delete All Sources API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred while deleting all documents."}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
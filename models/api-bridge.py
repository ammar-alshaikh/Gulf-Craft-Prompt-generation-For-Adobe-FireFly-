#!/usr/bin/env python3
"""
API Bridge for Gulf Craft Prompt Generator
This script provides a simple HTTP server that the Node.js backend can call to get AI-generated prompts.
"""

import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import logging

# Import the prompt generator directly
import importlib.util
spec = importlib.util.spec_from_file_location("prompt_generator", os.path.join(os.path.dirname(__file__), "prompt-generator.py"))
prompt_generator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_generator_module)
prompt_generator = prompt_generator_module.prompt_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptAPIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for prompt generation."""
        try:
            # Parse request
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Extract user input
            user_input = request_data.get('input', '')
            max_length = request_data.get('max_length', 200)
            
            if not user_input:
                self.send_error_response(400, "Missing 'input' parameter")
                return
            
            # Generate prompt
            result = prompt_generator.generate_prompt(user_input, max_length)
            
            # Send response
            self.send_success_response(result)
            
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            self.send_error_response(500, f"Internal server error: {str(e)}")
    
    def do_GET(self):
        """Handle GET requests for health check."""
        if self.path == '/health':
            self.send_success_response({
                "status": "healthy",
                "model_loaded": prompt_generator.is_loaded
            })
        else:
            self.send_error_response(404, "Not found")
    
    def send_success_response(self, data):
        """Send a successful JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = json.dumps(data, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))
    
    def send_error_response(self, status_code, message):
        """Send an error JSON response."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = json.dumps({
            "error": message,
            "status_code": status_code
        }, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")

def start_api_server(port=5001, model_path=None):
    """Start the API server."""
    try:
        # Load the model
        logger.info("Loading AI model...")
        logger.info(f"Model path: {model_path}")
        
        # Convert to absolute path if provided
        if model_path:
            model_path = os.path.abspath(model_path)
            logger.info(f"Absolute model path: {model_path}")
        
        prompt_generator.load_model(model_path)
        
        if not prompt_generator.is_loaded:
            logger.error("Failed to load model. Server will start but prompt generation will fail.")
        
        # Start server
        server_address = ('', port)
        httpd = HTTPServer(server_address, PromptAPIHandler)
        logger.info(f"Starting API server on port {port}")
        logger.info(f"Health check: http://localhost:{port}/health")
        logger.info("Press Ctrl+C to stop the server")
        
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        httpd.shutdown()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gulf Craft Prompt Generator API Server')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    parser.add_argument('--model-path', type=str, help='Path to the model files')
    
    args = parser.parse_args()
    
    start_api_server(args.port, args.model_path) 
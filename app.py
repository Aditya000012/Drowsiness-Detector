from flask import Flask, render_template, jsonify
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

@app.route('/api/config')
def config():
    return jsonify({
        'supabaseUrl': os.environ.get('SUPABASE_URL', ''),
        'supabaseAnonKey': os.environ.get('SUPABASE_ANON_KEY', '')
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

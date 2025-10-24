from flask import Flask, render_template

# Initialize a new Flask app just for testing
app = Flask(__name__)

@app.route('/')
def test_page():
    """
    This is the only page this server knows about.
    It will render the test_lottie.html file from your templates folder.
    """
    print("✅ Serving the Lottie test page.")
    return render_template('test_lottie.html')

if __name__ == '__main__':
    print("🚀 Starting Lottie test server...")
    print("➡️ Open this address in your browser: http://127.0.0.1:5500")
    app.run(port=5500, debug=True)

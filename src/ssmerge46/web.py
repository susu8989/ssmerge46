from threading import Thread

from flask import Flask

app = Flask("")


@app.route("/")
def root():
    """ヘルスチェック用."""
    return "🤗"


def run(port: int):
    t = Thread(target=app.run, kwargs=dict(debug=False, host="0.0.0.0", port=port))
    t.start()

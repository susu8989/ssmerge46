"""Flask."""
from threading import Thread

from flask import Flask

app = Flask("")


@app.route("/")
def root():
    """ヘルスチェック用."""
    return "🤗"


def run(port: int):
    """WEBサーバーを別スレッドで起動する.

    Args:
        port (int): 使用するポート.
    """
    t = Thread(target=app.run, kwargs={"debug": False, "host": "0.0.0.0", "port": port})
    t.start()

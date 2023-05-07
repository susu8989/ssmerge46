import io
import os

import cv2
import numpy as np
from discord import ChannelType, Client, File, Intents, Message

from ssmerge46.umamerge import stitch

USAGE = """```
スキル表示画面・因子表示画面を分割して撮影した複数のスクリーンショットを1枚に結合します。

使い方:
    - このbotが所属しているチャンネルで '/ssm' と入力し、合成したい画像を *一度に* 添付してください。
    - このbotにDMとして直接送信することも可能で、その場合コマンド入力は不要です。

注意:
    - 各画像は解像度を一致させてください。
    - 結合位置の検出のため、各画像はスキル表示部分1～1.5個分の重複部分（のりしろ）を作ってください。
    - 結合位置が見つからなかった場合は、単純に真下に結合します。
    - Botのサーバーダウンでたまに応答しないことがありますが悪しからず。
```"""

# Intents
intents = Intents.default()
intents.dm_messages = True
intents.guild_messages = True
intents.message_content = True

# Client
client = Client(intents=intents)


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message: Message):
    if message.author == client.user:
        return

    if message.channel.type == ChannelType.private or message.content.startswith(
        "/ssm"
    ):
        try:
            attachments = message.attachments
            if len(attachments) == 0:
                await message.channel.send(USAGE)
                return

            if len(attachments) < 2:
                await message.channel.send(
                    """```2枚以上の画像を読み込んでください。対応形式は png, jpg です。```"""
                )
                return

            imgs = []
            for attachment in attachments:
                filename = attachment.filename
                if (
                    filename.endswith(".png")
                    or filename.endswith(".jpg")
                    or filename.endswith(".jpeg")
                ):
                    buf = await attachment.read()
                    arr = np.frombuffer(buf, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    imgs.append(img)

            # merge
            stitched = stitch(imgs)

            success, encoded = cv2.imencode(".png", stitched)
            if not success:
                await message.channel.send("""```出力画像のエンコードに失敗しました。```""")
            data = io.BytesIO(encoded)
            file = file = File(data, "out.png")

            if message.channel.type == ChannelType.private:
                msg = """(´・ω・｀) できたよお兄ちゃん！"""
                await message.channel.send(msg, file=file)
                return

            await message.channel.send(file=file)
            return
        except Exception as e:
            await message.channel.send(f"An error ocurred: {e}")


if __name__ == "__main__":
    token = os.getenv("TOKEN")
    if not token:
        raise ValueError("'TOKEN' not defined in environment variables.")

    client.run(token)

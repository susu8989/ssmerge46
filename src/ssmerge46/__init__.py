import io
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import toml
from discord import ChannelType, Client, File, Intents, Message

from cv2wrap.image import BgrImage
from ssmerge46 import config
from ssmerge46.imgproc import stack_left_to_right, stack_top_to_bot
from ssmerge46.umamerge import ScrollStitcher

logger = logging.getLogger("discord")


# Intents
intents = Intents.default()
intents.dm_messages = True
intents.guild_messages = True
intents.message_content = True

# Client
client = Client(intents=intents)

stitcher = ScrollStitcher()


def _get_version() -> str:
    try:
        d = toml.load(Path(__file__).parent.parent.parent / "pyproject.toml")
        return d["tool"]["poetry"]["version"]
    except Exception as e:
        logger.warn(e, exc_info=True)
        return ""


version = _get_version()

USAGE = f"""```
# SSマージ四郎 ver.{version}
    スキル表示画面・因子表示画面を分割して撮影した複数のスクリーンショットを1枚に結合します。

## 使い方
    - このbotが所属しているチャンネルで '/ssm' と入力し、合成したい画像を *一度に* 添付してください。
    - このbotにDMとして直接送信することも可能で、その場合コマンド入力は不要です。

## 注意
    - 各画像は解像度を一致させてください。
    - 結合位置の検出のため、各画像はスキル表示部分1～1.5個分の重複部分（のりしろ）を作ってください。
    - 結合位置が見つからなかった場合はエラーになります。のりしろを増やしてみてください。
    - Botのサーバーダウンでたまに応答しないことがあります。
```"""


@client.event
async def on_ready():
    print(f"We have logged in as {client.user} ({version})")


@client.event
async def on_message(message: Message):
    if message.author == client.user:
        return

    now = datetime.now()
    content = message.content

    if message.channel.type == ChannelType.private or content.startswith("/ssm"):
        attachments = message.attachments
        if len(attachments) == 0:
            await message.channel.send(USAGE)
            return
        if len(attachments) < 2:
            await message.channel.send("2枚以上の画像を送信してください。対応形式は png / jpg です。")
            return

        try:
            imgs: List[BgrImage] = []
            for attachment in attachments:
                filename = attachment.filename
                if (
                    filename.endswith(".png")
                    or filename.endswith(".jpg")
                    or filename.endswith(".jpeg")
                    or filename.endswith(".webp")
                ):
                    buf = await attachment.read()
                    arr = np.frombuffer(buf, np.uint8)
                    img = BgrImage.decode(arr)
                    if img.px > config.MAX_RESOLUTION:
                        scale = config.MAX_RESOLUTION / img.px
                        img = img.scale_img(scale)
                    imgs.append(img)
                else:
                    logger.warn(f"Invalid file type : {filename}")
                    raise ValueError(f"非対応の画像形式です。対応形式は png / jpg です。 : {filename}")

            if content.startswith("/ssmh"):
                merged = stack_left_to_right(imgs)
            elif content.startswith("/ssmv"):
                merged = stack_top_to_bot(imgs)
            else:
                # merge
                merged = stitcher.stitch(imgs)
            logger.info("%s", merged.wh)
            success, encoded = cv2.imencode(".png", merged)
            if not success:
                await message.channel.send("出力画像のエンコードに失敗しました。")
            data = io.BytesIO(encoded)
            now_str = now.strftime("%Y%m%d%H%M%S%f")[:-3]
            file = file = File(data, f"{now_str}.png")

            msg = _get_random_msg()
            await message.channel.send(msg, file=file)
        except Exception as e:
            logger.warn(e, exc_info=True)
            await message.channel.send(f"[ERROR] {e}")
        finally:
            author = message.author
            logger.info(
                "[%s %s %s] %d %s",
                author.id,
                author.name,
                author.display_name,
                len(attachments),
                content,
            )
        return

    if config.MASANORI and random.random() < 0.01 and len(content) <= 32:
        if content.endswith("本命は？") or content.endswith("本命教えて"):
            msg = "フェーングロッテン"
        elif content == "まさのり":
            msg = "こ～んに～ちは～！！！"
        else:
            msg = ""

        if msg:
            await message.channel.send(msg)


WEIGHTS = {"(´・ω・｀) できたよお兄ちゃん！": 10, "フェーングロッテン": 5, "CR雅紀だよ！": 1, "": 84}


def _get_random_msg() -> Optional[str]:
    if not config.MASANORI:
        return None
    msg = random.choices(list(WEIGHTS.keys()), weights=list(WEIGHTS.values()))[0]
    return msg if msg else None


def start():
    client.run(config.TOKEN)


if __name__ == "__main__":
    start()

import io
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import toml
from discord import ChannelType, Client, File, Intents, Message
from toml import TomlDecodeError

from cv2wrap.image import BgrImage
from ssmerge46 import web
from ssmerge46.config import (
    FLASK_PORT,
    MASANORI,
    MASANORI_RATE,
    MAX_ATTACHMENTS,
    MAX_RESOLUTION,
    MAX_RESOLUTION_2,
    MAX_RESOLUTION_3,
    TOKEN,
)
from ssmerge46.exception import InvalidInputError, InvalidInputImageError
from ssmerge46.imgproc import (
    combine_left_to_right,
    combine_squarely,
    combine_top_to_bot,
    cv2_stitch,
)
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
    except (TypeError, TomlDecodeError) as e:
        logger.warning(e, exc_info=True)
        return ""


version = _get_version()

AVAILABLE_FILETYPES = ("png", "jpg", "jpeg", "webp")

USAGE = f"""```
# SSマージ四郎 ver.{version}
ウマ娘のゲーム画面のスクリーンショットを1枚に結合します。

## コマンド一覧
/ssm    ... ウマ娘スクロール画面の自動結合
  - スキル表示・因子表示画面の縦スクロールエリアを自動検出して縦に結合します。
  - その他の画面には基本的に使えません。
  - 画像の解像度は統一し、1～2割程度の重複部分（のりしろ）を作ってください。

/ssmh   ... ヨコ結合
  - 画像をヨコ方向に (= Horizontally) 単純に結合します。
  - 高さが異なる場合、1枚目を基準として2枚目以降は自動リサイズされます。

/ssmv   ... タテ結合
  - 画像をタテ方向に (= Vertically) 単純に結合します。
  - 幅が異なる場合、1枚目を基準として2枚目以降は自動リサイズされます。

/ssmz   ... タテヨコ結合
  - 画像を正方形に近くなるように並べて結合します。
  - 異なる場合、1枚目を基準として2枚目以降は自動リサイズされ、アスペクト比が異なる場合は黒い隙間ができます。

## 注意
- 対応形式は {' / '.join( AVAILABLE_FILETYPES)} です。
- 一度に送信可能な画像は 最大{MAX_ATTACHMENTS}枚 です。
- botにDMとして画像を送信することもでき、この場合 `/ssm` であればコマンド省略可能です。
- サーバーダウンでたまに応答しないことがあります。
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
        status = "OK"
        attachments = message.attachments
        attachments_len = len(attachments)
        if attachments_len == 0:
            await message.channel.send(USAGE)
            return

        logger.info(
            "[%s %s] [%s %s %s] request: %s",
            message.guild,
            message.channel,
            message.author.id,
            message.author.name,
            message.author.display_name,
            content,
        )

        imgs: list[BgrImage] = []
        try:
            if attachments_len < 2:
                raise InvalidInputImageError(
                    f"画像を2枚以上送信してください。対応形式は {' / '.join( AVAILABLE_FILETYPES)} です。"
                )
            if attachments_len > MAX_ATTACHMENTS:
                raise InvalidInputImageError(f"結合できる画像は 最大{MAX_ATTACHMENTS}枚 です。")

            if attachments_len <= 4:
                resolution_limit = MAX_RESOLUTION
            elif attachments_len <= 9:
                resolution_limit = MAX_RESOLUTION_2
            else:
                resolution_limit = MAX_RESOLUTION_3

            for attachment in attachments:
                filename = attachment.filename
                if filename.endswith(AVAILABLE_FILETYPES):
                    buf = await attachment.read()
                    arr = np.frombuffer(buf, np.uint8)
                    img = BgrImage.decode(arr)
                    if img.px > resolution_limit:
                        scale = resolution_limit / img.px
                        resized = img.scale_img(scale)
                        del img
                    else:
                        resized = img
                    imgs.append(resized)
                else:
                    raise InvalidInputError(
                        f"非対応の画像形式です。対応形式は {' / '.join( AVAILABLE_FILETYPES)}  です。 : {filename}"
                    )

            if content.startswith("/ssmh"):
                # Simple vstack
                merged = combine_left_to_right(imgs)
            elif content.startswith("/ssmv"):
                # Simple hstack
                merged = combine_top_to_bot(imgs)
            elif content.startswith("/ssmz"):
                # Simple hstack
                merged = combine_squarely(imgs)
            elif content.startswith("/ssms"):
                # OpenCV Sticher
                merged = cv2_stitch(imgs)
            else:
                # Scroll area Stitcher
                debug_enabled: bool = "debug" in content.split(" ")
                merged = stitcher.stitch(
                    imgs,
                    overlap_ratios=(0.2, 0.15, 0.1),
                    match_threshs=(0.85, 0.85, 0.9),
                    debug=debug_enabled,
                )

            logger.info("Merged. : %s", merged.wh)
            success, encoded = cv2.imencode(".png", merged)
            if not success:
                await message.channel.send("出力画像のエンコードに失敗しました。")
            data = io.BytesIO(encoded)  # type: ignore
            now_str = now.strftime("%Y%m%d%H%M%S%f")[:-3]
            file = file = File(data, f"{now_str}.png")

            msg = _get_random_msg()
            await message.channel.send(msg, file=file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            status = "ERROR"
            logger.warning(e, exc_info=True)
            await message.channel.send(f"[ERROR] {e}")
        finally:
            logger.info(
                "[%s %s] [%s %s %s] done: %s [%d] -> [%s]",
                message.guild,
                message.channel,
                message.author.id,
                message.author.name,
                message.author.display_name,
                content,
                attachments_len,
                status,
            )
        return

    if MASANORI and random.random() < MASANORI_RATE and len(content) <= 32:
        if content.endswith("本命は？") or content.endswith("本命教えて"):
            msg = "フェーングロッテン"
        elif content == "まさのり":
            msg = "こ～んに～ちは～！！！"
        elif content == "フェーングロッテン":
            msg = "本命は？"
        else:
            msg = ""

        if msg:
            await message.channel.send(msg)


WEIGHTS = {"(´・ω・｀) できたよお兄ちゃん！": 10, "フェーングロッテン": 5, "CR雅紀だよ！": 1, "": 84}


def _get_random_msg() -> Optional[str]:
    if not MASANORI:
        return None
    msg = random.choices(list(WEIGHTS.keys()), weights=list(WEIGHTS.values()))[0]
    return msg if msg else None


def start():
    web.run(FLASK_PORT)
    client.run(TOKEN)


if __name__ == "__main__":
    start()

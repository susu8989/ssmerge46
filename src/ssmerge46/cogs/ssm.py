import io
import logging
import random
import time
from datetime import datetime

import cv2
import numpy as np
from discord import ChannelType, File, Interaction, Message, app_commands
from discord.ext import commands

from cv2wrap.image import BgrImage
from ssmerge46.config import (
    GUILD,
    MASANORI,
    MASANORI_RATE,
    MAX_ATTACHMENTS,
    MAX_RESOLUTION,
    MAX_RESOLUTION_2,
    MAX_RESOLUTION_3,
    VERSION,
)
from ssmerge46.exception import (
    BotProcessingError,
    ImageProcessingError,
    InvalidInputError,
    InvalidInputImageError,
)
from ssmerge46.imgproc import (
    combine_left_to_right,
    combine_squarely,
    combine_top_to_bot,
    cv2_stitch,
)
from ssmerge46.umamerge import ScrollStitcher

log = logging.getLogger("discord")

AVAILABLE_FILETYPES = set(["png", "jpeg", "webp"])
AVAILABLE_MIMETYPES = set(f"image/{t}" for t in AVAILABLE_FILETYPES)

USAGE = f"""```
# SSマージ四郎 ver.{VERSION}
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

WEIGHTS = {
    "(´・ω・｀) できたよお兄ちゃん！": 10,
    "フェーングロッテン": 5,
    "CR雅紀だよ！": 1,
    "": 84,
}


def _generate_success_text() -> str:
    if not MASANORI:
        return ""
    text = random.choices(list(WEIGHTS.keys()), weights=list(WEIGHTS.values()))[0]
    return text if text else ""


class SsmCog(commands.Cog, name="ssm"):

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.stitcher = ScrollStitcher()

        self.bot.tree.add_command(
            app_commands.ContextMenu(name="スクロール結合 (ssm)", callback=self.context_ssm)
        )
        self.bot.tree.add_command(
            app_commands.ContextMenu(name="ヨコ結合 (ssmh)", callback=self.context_ssmh)
        )
        self.bot.tree.add_command(
            app_commands.ContextMenu(name="タテ結合 (ssmv)", callback=self.context_ssmv)
        )
        self.bot.tree.add_command(
            app_commands.ContextMenu(name="タテヨコ結合 (ssmz)", callback=self.context_ssmz)
        )
        self.bot.tree.add_command(
            app_commands.ContextMenu(name="結合結果を削除", callback=self.context_remove)
        )

    @commands.command(
        name="ssm",
        description="ウマ娘アプリのスキル表示・因子表示画面中の縦スクロールエリアを検出して1枚に結合します。",
    )
    @commands.cooldown(1, 1, commands.BucketType.member)
    async def ssm(self, ctx: commands.Context) -> None:
        await self._handle_message_command(ctx.message, "ssm")

    @commands.command(
        name="ssmh",
        description="添付した画像をヨコ方向 (左→右) に結合します。1枚目を基準として2枚目以降は自動リサイズされます。",
    )
    @commands.cooldown(1, 1, commands.BucketType.member)
    async def ssmh(self, ctx: commands.Context) -> None:
        await self._handle_message_command(ctx.message, "ssmh")

    @commands.command(
        name="ssmv",
        description="添付した画像をタテ方向 (上→下) に結合します。1枚目を基準として2枚目以降は自動リサイズされます。",
    )
    @commands.cooldown(1, 1, commands.BucketType.member)
    async def ssmv(self, ctx: commands.Context) -> None:
        await self._handle_message_command(ctx.message, "ssmv")

    @commands.command(
        name="ssmz",
        description="添付した画像を正方形に近くなるように並べて結合します。1枚目を基準として2枚目以降は自動リサイズされます。",
    )
    @commands.cooldown(1, 1, commands.BucketType.member)
    async def ssmz(self, ctx: commands.Context) -> None:
        await self._handle_message_command(ctx.message, "ssmz")

    @app_commands.checks.has_permissions(ban_members=True)
    @app_commands.guilds(GUILD)
    async def context_ssm(self, interaction: Interaction, message: Message) -> None:
        await self._handle_context_command(interaction, message, "ssm")

    @app_commands.checks.has_permissions(ban_members=True)
    @app_commands.guilds(GUILD)
    async def context_ssmh(self, interaction: Interaction, message: Message) -> None:
        await self._handle_context_command(interaction, message, "ssmh")

    @app_commands.checks.has_permissions(ban_members=True)
    @app_commands.guilds(GUILD)
    async def context_ssmv(self, interaction: Interaction, message: Message) -> None:
        await self._handle_context_command(interaction, message, "ssmv")

    @app_commands.checks.has_permissions(ban_members=True)
    @app_commands.guilds(GUILD)
    async def context_ssmz(self, interaction: Interaction, message: Message) -> None:
        await self._handle_context_command(interaction, message, "ssmz")

    @app_commands.checks.has_permissions(ban_members=True)
    @app_commands.guilds(GUILD)
    async def context_remove(self, interaction: Interaction, message: Message) -> None:
        if message.author != self.bot.user:
            log.warning("Not bot message : author=%s", message.author)
            await interaction.response.send_message(
                "[ERROR] このBotの実行結果ではありません。",
                ephemeral=True,
            )
            return
        if interaction.user not in message.mentions and (
            message.interaction and interaction.user != message.interaction.user
        ):
            log.warning(
                "User not matched : user=%s, mentions=%s", interaction.user, message.mentions
            )
            await interaction.response.send_message(
                "[ERROR] 他のユーザーの実行結果は削除できません。",
                ephemeral=True,
            )
            return

        await message.delete()
        created_at = message.created_at.strftime("%m/%d %H:%M")
        await interaction.response.send_message(
            f"{created_at} の結果を削除しました。",
            ephemeral=True,
        )
        log.info(
            "[%s %s] [%s %s %s] context#%s %s",
            message.guild or "None",
            message.channel or "None",
            message.author.id,
            message.author.name,
            message.author.display_name,
            "remove",
            created_at,
        )
        return

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        log.info("%s on ready!", self.__class__.__name__)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        content = message.content
        if message.channel.type == ChannelType.private:
            # DMの場合は空メッセージでも /ssm として動作
            if not content:
                await self._handle_message_command(message, "ssm")
                return

        if MASANORI and random.random() < MASANORI_RATE and len(content) <= 32:
            text = self._gen_random_message(content)
            if text:
                await message.channel.send(text)
            return

    async def _handle_context_command(
        self, interaction: Interaction, message: Message, command: str
    ) -> None:
        content = message.content
        before = time.monotonic()
        status = "OK"
        await interaction.response.defer(thinking=True)
        try:
            file = await self._proc_images(message, command, "debug" in content.split(" "))
            text = _generate_success_text()
            await interaction.followup.send(content=text, file=file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if isinstance(e, BotProcessingError):
                log.warning(e, exc_info=True)
            else:
                log.error(e, exc_info=True)
            status = "ERROR"
            await interaction.followup.send(content=f"[ERROR] {e}")
        finally:
            msec = (time.monotonic() - before) * 1000
            log.info(
                "[%s %s] [%s %s %s] context#%s %d -> [%s %dms]",
                message.guild or "None",
                message.channel or "None",
                message.author.id,
                message.author.name,
                message.author.display_name,
                command,
                len(message.attachments),
                status,
                msec,
            )
        return

    async def _handle_message_command(self, message: Message, command: str) -> None:
        if message.author == self.bot.user:
            # DO NOTHING
            return

        if len(message.attachments) == 0:
            await message.channel.send(USAGE)
            return

        content = message.content
        before = time.monotonic()
        status = "OK"

        try:
            file = await self._proc_images(message, command, "debug" in content.split(" "))
            text = _generate_success_text()
            await message.channel.send(f"{message.author.mention} {command} {text}", file=file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if isinstance(e, BotProcessingError):
                log.warning(e, exc_info=True)
            else:
                log.error(e, exc_info=True)
            status = "ERROR"
            await message.channel.send(f"{message.author.mention} [ERROR] {e}")
        finally:
            msec = (time.monotonic() - before) * 1000
            log.info(
                "[%s %s] [%s %s %s] %s %d -> [%s %dms]",
                message.guild or "None",
                message.channel or "None",
                message.author.id,
                message.author.name,
                message.author.display_name,
                content,
                len(message.attachments),
                status,
                msec,
            )
        return

    # async def _handle_command(self, ctx: commands.Context) -> None:
    #     if len(imgs) == 0 or not ctx.command:
    #         await ctx.send(USAGE)
    #         return

    #     before = time.monotonic()
    #     message = await ctx.send("処理中です...")
    #     file = await self._proc_images(ctx.message, ctx.command.name, debug)
    #     msec = (time.monotonic() - before) * 1000

    #     text = _generate_success_text()
    #     await message.edit(content=f"{text} `{int(msec)} ms`", attachments=[file])

    async def _proc_images(self, message: Message, command: str, debug: bool = False) -> File:
        now = datetime.now()
        attachments = message.attachments
        attachments_len = len(attachments)
        if attachments_len < 2:
            raise InvalidInputImageError(
                f"2枚以上の画像が必要です。対応形式は {' / '.join(AVAILABLE_FILETYPES)} です。"
            )
        if attachments_len > MAX_ATTACHMENTS:
            raise InvalidInputImageError(f"結合できる画像は、最大{MAX_ATTACHMENTS}枚です。")

        if attachments_len <= 4:
            resolution_limit = MAX_RESOLUTION
        elif attachments_len <= 9:
            resolution_limit = MAX_RESOLUTION_2
        else:
            resolution_limit = MAX_RESOLUTION_3

        imgs: list[BgrImage] = []
        for attachment in attachments:
            content_type = attachment.content_type
            if content_type and content_type in AVAILABLE_MIMETYPES:
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
                    f"非対応の画像形式です。対応形式は {' / '.join(AVAILABLE_FILETYPES)}  です。 : {attachment.filename}"
                )

        if command == "ssmh":
            # Simple vstack
            merged = combine_left_to_right(imgs)
        elif command == ("ssmv"):
            # Simple hstack
            merged = combine_top_to_bot(imgs)
        elif command == ("ssmz"):
            # Simple hstack
            merged = combine_squarely(imgs)
        elif command == ("ssms"):
            # OpenCV Sticher
            merged = cv2_stitch(imgs)
        else:
            # Scroll area Stitcher
            merged = self.stitcher.stitch(
                imgs,
                overlap_ratios=(0.2, 0.15, 0.1),
                match_threshs=(0.85, 0.85, 0.9),
                debug=debug,
            )

        success, encoded = cv2.imencode(".png", merged)
        if not success:
            raise ImageProcessingError("出力画像のエンコードに失敗しました。")
        data = io.BytesIO(encoded)  # type: ignore
        now_str = now.strftime("%Y%m%d%H%M%S%f")[:-3]
        file = File(data, f"{now_str}.png")
        return file

    def _gen_random_message(self, content: str) -> str:
        if content.endswith("本命は？") or content.endswith("本命教えて"):
            return "フェーングロッテン"
        if content == "まさのり":
            return "こ～んに～ちは～！！！"
        if content == "フェーングロッテン":
            return "本命は？"
        return ""


async def setup(bot: commands.Bot):
    await bot.add_cog(SsmCog(bot))

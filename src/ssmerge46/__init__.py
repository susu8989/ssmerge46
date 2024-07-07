import logging

import discord
from discord import Intents
from discord.ext import commands

from ssmerge46.config import GUILD, TOKEN, VERSION

INITAL_EXTENSTIONS = [
    "cogs.ssm",
    "cogs.command_error",
]
PREFIX = "/"
INTENTS = Intents.default()
INTENTS.dm_messages = True
INTENTS.guild_messages = True
INTENTS.message_content = True
log = logging.getLogger("discord")


class MyBot(commands.Bot):
    """SSmerge46 bot."""

    def __init__(
        self,
        command_prefix=PREFIX,
        help_command=None,
        description="ウマ娘のゲーム画面のスクリーンショットを1枚に結合します.",
        intents=INTENTS,
    ):
        super().__init__(
            command_prefix=command_prefix,
            help_command=help_command,
            description=description,
            intents=intents,
        )

    async def setup_hook(self) -> None:
        await self.load_cogs()
        await self.tree.sync(guild=GUILD)
        # await self.tree.sync()

    async def on_ready(self) -> None:
        log.info("We have logged in as %s (%s)", self.user, VERSION)

        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching, name=f"{self.command_prefix}help"
            )
        )

    async def load_cogs(self) -> None:
        for cog in INITAL_EXTENSTIONS:
            log.info("Loading Cog : %s", cog)
            await self.load_extension(cog)


if __name__ == "__main__":
    bot = MyBot()

    # @app_commands.guilds(GUILD)
    # @bot.command(name="upload_many")
    # async def upload_many(
    #     ctx,
    #     *remaining: discord.Attachment,
    # ):
    #     files: list[str] = []
    #     files.extend(a.url for a in remaining)
    #     await ctx.send(f'You uploaded: {" ".join(files)}')

    bot.run(TOKEN)

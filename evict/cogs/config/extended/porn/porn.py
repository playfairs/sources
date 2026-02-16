from discord.ext.commands import group, has_permissions, is_nsfw
from core.client.context import Context
from tools import CompositeMetaClass, MixinMeta
import asyncio
from typing import Dict, List
import discord
import config
import aiohttp

from discord import Embed
from discord.ext import tasks
from random import choice
from datetime import datetime, timezone
import tempfile
import aiohttp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import time
import os
from logging import getLogger

log = getLogger("evict/porn")

class Porn(MixinMeta, metaclass=CompositeMetaClass):
    """
    Porn search functionality.
    """

    def __init__(self, bot):
        try:
            super().__init__(bot)
            self.bot = bot
            self.name = "Porn Search"
            self.session = aiohttp.ClientSession()
        except Exception as e:
            return

    async def cog_load(self) -> None:
        """Initialize when cog loads"""
        try:
            self.send_content.start()
            await super().cog_load()
        except Exception as e:
            return

    async def cog_unload(self) -> None:
        """Cleanup when cog unloads"""
        try:
            self.send_content.cancel()
            if hasattr(self, 'session'):
                await self.session.close()
        except Exception as e:
            return
        
        try:
            await super().cog_unload()
        except Exception as e:
            return

    @tasks.loop(minutes=3)
    async def send_content(self):
        """Send content every minute to configured channels"""
        try:
            configs = await self.bot.db.fetch(
                """
                SELECT guild_id, channel_id, webhook_id, webhook_token, spoiler 
                FROM porn.config
                """
            )

            if not configs:
                return

            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.adultdatalink.com/redgifs/trending?parameter=gif") as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to fetch content: Status {resp.status}")
                    
                    data = await resp.json()
                    if not data.get("items"):
                        raise Exception("No content available in API response")

                content = choice(data["items"])
                
                temp_path = f"/tmp/temp_{int(time.time())}.mp4"
                async with session.get(content["urls"]["vthumbnail"]) as resp:
                    if resp.status == 200:
                        content_data = await resp.read()
                        with open(temp_path, 'wb') as f:
                            f.write(content_data)
                    else:
                        temp_path = None

                    for config in configs:
                        try:
                            guild = self.bot.get_guild(config['guild_id'])
                            if not guild:
                                continue

                            channel = guild.get_channel(config['channel_id'])
                            if not channel:
                                continue

                            if not channel.is_nsfw():
                                webhook = discord.Webhook.from_url(
                                    f"https://discord.com/api/webhooks/{config['webhook_id']}/{config['webhook_token']}", 
                                    session=session
                                )
                                await webhook.send(
                                    content="⚠️ This channel is no longer marked as NSFW. Content delivery has been paused until the channel is marked as NSFW again."
                                )
                                continue

                            webhook = discord.Webhook.from_url(
                                f"https://discord.com/api/webhooks/{config['webhook_id']}/{config['webhook_token']}", 
                                session=session
                            )

                            if guild.icon:
                                await webhook.edit(avatar=await guild.icon.read())

                            await webhook.send(file=discord.File(temp_path))
                                                  #spoiler=config['spoiler']))
                            await asyncio.sleep(2)

                        except Exception as e:
                            pass

            os.remove(temp_path)

        except Exception as e:
            return

    @send_content.before_loop
    async def before_send_content(self):
        """Wait for bot to be ready before starting the loop"""
        await self.bot.wait_until_ready()

    @send_content.error
    async def send_content_error(self, error):
        """Handle any errors in the task"""

    @send_content.after_loop
    async def after_send_content(self):
        """Log when the task stops"""

    @is_nsfw()
    @group(invoke_without_command=True)
    async def porn(self, ctx: Context):
        """Porn search commands"""
        return await ctx.send_help(ctx.command)

    @porn.command(name="add")
    @has_permissions(manage_guild=True)
    async def porn_add(self, ctx: Context, channel: discord.TextChannel):
        """Set up an Porn search channel"""
        if not channel.is_nsfw():
            return await ctx.warn("Channel must be marked as NSFW")

        try:
            webhook = await channel.create_webhook(
                name=f"{ctx.guild.name} Porn Search",
                reason="Automated porn search webhook creation"
            )

            await self.bot.db.execute(
                """
                INSERT INTO porn.config (guild_id, channel_id, webhook_id, webhook_token)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (guild_id) 
                DO UPDATE SET 
                    channel_id = $2,
                    webhook_id = $3,
                    webhook_token = $4
                """,
                ctx.guild.id,
                channel.id,
                webhook.id,
                webhook.token
            )
            
            await ctx.approve(f"Porn search channel set to {channel.mention}")
        except Exception as e:
            await ctx.warn(f"Failed to set up porn search: {e}")

    @porn.command(name="remove", aliases=["delete", "del"])
    @has_permissions(manage_guild=True)
    async def porn_remove(self, ctx: Context):
        """Remove the PORN search channel"""
        config = await self.bot.db.fetchrow(
            """
            DELETE FROM porn.config 
            WHERE guild_id = $1
            RETURNING webhook_id, channel_id
            """,
            ctx.guild.id
        )

        if not config:
            return await ctx.warn("No porn search channel configured!")

        try:
            channel = ctx.guild.get_channel(config['channel_id'])
            if channel:
                webhooks = await channel.webhooks()
                webhook = discord.utils.get(webhooks, id=config['webhook_id'])
                if webhook:
                    await webhook.delete()
        except:
            pass

        await ctx.approve("Porn search configuration removed")

    @porn.command(name="view")
    @has_permissions(manage_guild=True)
    async def porn_view(self, ctx: Context):
        """View current porn search configuration"""

        config = await self.bot.db.fetchrow(
            """
            SELECT channel_id FROM porn.config
            WHERE guild_id = $1
            """,
            ctx.guild.id
        )

        if not config:
            return await ctx.warn("No porn search channel configured!")

        channel = ctx.guild.get_channel(config['channel_id'])
        if not channel:
            return await ctx.warn("Configured channel no longer exists!")

        await ctx.approve(f"Porn search channel: {channel.mention}")

    @porn.command(name="trending", aliases=["tags", "popular"])
    async def porn_trending(self, ctx: Context):
        """View trending search tags"""
        if not ctx.channel.is_nsfw():
            return await ctx.warn("Channel must be marked as NSFW")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.adultdatalink.com/redgifs/tags?trending=true") as resp:
                    if resp.status != 200:
                        return await ctx.warn("Failed to fetch trending tags")
                    
                    data = await resp.json()

            if not data.get("items"):
                return await ctx.warn("No trending tags found")

            embed = Embed(
                title="Trending Porn Tags",
                color=ctx.color
            )

            formatted_tags = []
            for item in data["items"][:10]:  
                count = f'{item["count"]:,}' 
                formatted_tags.append(f"`{item['name']}` ({count} searches)")

            embed.description = "\n".join(formatted_tags)
            embed.set_footer(text="Updated every hour")

            await ctx.send(embed=embed)

        except Exception as e:
            await ctx.warn(f"Error fetching trending tags: {e}")

    @porn.command(name="spoiler")
    @has_permissions(manage_guild=True)
    async def porn_spoiler(self, ctx: Context):
        """Toggle spoiler mode for porn videos"""
        config = await self.bot.db.fetchrow(
            """
            UPDATE porn.config 
            SET spoiler = NOT spoiler 
            WHERE guild_id = $1 
            RETURNING spoiler
            """,
            ctx.guild.id
        )

        if not config:
            return await ctx.warn("No porn channel configured!")

        status = "enabled" if config['spoiler'] else "disabled"
        await ctx.approve(f"Spoiler mode {status}")

    @porn.command(name="search")
    async def porn_search(self, ctx: Context, *, tag: str):
        """Search for porn content by tag"""
        if not ctx.channel.is_nsfw():
            return await ctx.warn("Channel must be marked as NSFW")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.adultdatalink.com/redgifs/search?q={tag}") as resp:
                    if resp.status != 200:
                        return await ctx.warn("Failed to fetch search results")
                    
                    data = await resp.json()

            if not data.get("items"):
                return await ctx.warn(f"No results found for tag `{tag}`")

            content = choice(data["items"])
            await ctx.send(content["urls"]["hd"])

        except Exception as e:
            await ctx.warn(f"Error searching for content: {e}")

    @porn.command(name="random")
    async def porn_random(self, ctx: Context):
        """Get random porn content"""
        if not ctx.channel.is_nsfw():
            return await ctx.warn("Channel must be marked as NSFW")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.adultdatalink.com/redgifs/random") as resp:
                    if resp.status != 200:
                        return await ctx.warn("Failed to fetch random content")
                    
                    data = await resp.json()

            if not data.get("urls"):
                return await ctx.warn("No content available")

            await ctx.send(data["urls"]["hd"])

        except Exception as e:
            await ctx.warn(f"Error fetching random content: {e}")
import os
import discord
from datetime import datetime, timezone
import logging
import platform
import psutil
import sys
import config

from main import Evict
from discord.ext.commands import Cog
from posthog import Posthog
from discord.app_commands import Command as AppCommand
from discord.ext.commands import HybridCommand

posthog_cloud = Posthog(
    "phc_9CO68Be49weXo25XXmg0CY7wwFLl2HMzW5qoXWdH1J8", 
    host="https://us.i.posthog.com"
)

posthog_instances = [posthog_cloud]

if getattr(config.POSTHOG, 'TRACK_SELF', False):
    posthog_selfhost = Posthog(
        "phc_urcPFFBU0iLLTKBtmEfViMen8uFC8DiVnJorIcGmBBL",
        host="https://hog.evict.bot"
    )
    posthog_instances.append(posthog_selfhost)

for ph in posthog_instances:
    ph.batch_size = 50
    ph.max_queue_size = 10000
    ph.thread_count = 1
    ph.flush_interval = 0.5
    ph.debug = False

log = logging.getLogger("evict/posthog").setLevel(logging.ERROR)


class Hog(Cog):
    def __init__(self, bot: Evict):
        self.bot = bot
        self.description = "Posthog Analytics for Evict"
        self.bot.after_invoke(self.after_invoke)
        self.error_cooldown = {}

    async def _send_analytics(self, ph, distinct_id, event, properties, groups=None):
        current_time = datetime.now(timezone.utc)
        error_key = f"{ph.host}_{event}"
        
        if error_key in self.error_cooldown:
            if (current_time - self.error_cooldown[error_key]).total_seconds() < 300:
                return
            
        try:
            ph.capture(
                distinct_id=distinct_id,
                event=event,
                properties=properties,
                groups=groups
            )
            ph.flush()
        except Exception as e:
            self.error_cooldown[error_key] = current_time
            log.warning(f"Analytics failed for {ph.host} - {event}: Will retry in 5 minutes")

    async def after_invoke(self, ctx):
        properties = {
            "command_name": ctx.command.qualified_name,
            "command_category": ctx.command.cog_name,
            "is_owner": await self.bot.is_owner(ctx.author),
            "channel_type": str(ctx.channel.type),
            "message_length": len(ctx.message.content),
            "has_attachments": bool(ctx.message.attachments),
            "has_embeds": bool(ctx.message.embeds),
            "prefix_used": ctx.prefix,
            "command_latency": round(self.bot.latency * 1000, 2),
            "command_args": bool(ctx.args[2:]),  
            "failed": ctx.command_failed if hasattr(ctx, 'command_failed') else False,
            "cpu_percent": psutil.cpu_percent(),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
            "thread_count": psutil.Process().num_threads(),
            "python_version": sys.version,
            "platform": platform.platform()
        }
        
        groups = {}
        
        if ctx.guild:
            groups["guild"] = str(ctx.guild.id)
            
            if not ctx.guild.chunked:
                try:
                    await ctx.guild.chunk()
                except Exception as e:
                    log.error(f"Failed to chunk guild {ctx.guild.id}: {e}")

            properties.update({
                "guild_member_count": getattr(ctx.guild, 'member_count', 0),
                "guild_channel_count": len(getattr(ctx.guild, 'channels', [])),
                "guild_role_count": len(getattr(ctx.guild, 'roles', [])),
                "user_permissions": [perm[0] for perm in ctx.channel.permissions_for(ctx.author) if perm[1]] if ctx.channel else [],
                "is_guild_owner": ctx.author.id == getattr(ctx.guild, 'owner_id', None),
                "user_roles": [str(role.id) for role in getattr(ctx.author, 'roles', [])] if hasattr(ctx.author, 'roles') else [],
                "channel_nsfw": ctx.channel.is_nsfw() if hasattr(ctx.channel, 'is_nsfw') else False,
                "guild_chunked": ctx.guild.chunked 
            })

        selfhost_properties = {
            **properties,
            "system_info": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
                "thread_count": psutil.Process().num_threads()
            },
            "command_timing": {
                "started_at": ctx.message.created_at.isoformat(),
                "processing_time": round((datetime.now(timezone.utc) - ctx.message.created_at).total_seconds() * 1000, 2)
            },
            "message_details": {
                "content_length": len(ctx.message.content),
                "has_mentions": bool(ctx.message.mentions),
                "has_role_mentions": bool(ctx.message.role_mentions),
                "has_channel_mentions": bool(ctx.message.channel_mentions)
            },
            "user_details": {
                "name": str(ctx.author),
                "created_at": ctx.author.created_at.isoformat(),
                "bot": ctx.author.bot,
                "system": ctx.author.system
            }
        }

        if ctx.guild:
            selfhost_properties.update({
                "guild_details": {
                    "name": ctx.guild.name,
                    "created_at": ctx.guild.created_at.isoformat(),
                    "member_count": ctx.guild.member_count,
                    "channel_count": len(ctx.guild.channels),
                    "role_count": len(ctx.guild.roles),
                    "boost_level": ctx.guild.premium_tier,
                    "features": list(ctx.guild.features),
                    "emoji_count": len(ctx.guild.emojis),
                    "sticker_count": len(ctx.guild.stickers),
                    "voice_states": len(ctx.guild._voice_states)
                }
            })

        for ph in posthog_instances:
            properties_to_use = selfhost_properties if ph.host == "https://hog.evict.bot" else properties
            await self._send_analytics(ph, str(ctx.author.id), "command_invoked", properties_to_use, groups)

    # @Cog.listener("on_guild_join")
    # async def on_guild_join(self, guild: discord.Guild):
    #     selfhost_properties = {
    #         "name": guild.name,
    #         "member_count": guild.member_count,
    #         "channel_count": len(guild.channels),
    #         "role_count": len(guild.roles),
    #         "created_at": guild.created_at.isoformat(),
    #         "features": list(guild.features),
    #         "verification_level": str(guild.verification_level),
    #         "premium_tier": guild.premium_tier,
    #         "premium_subscription_count": guild.premium_subscription_count,
    #         "owner_id": str(guild.owner_id),
    #         "has_bot_manager_role": any(role.name.lower() == "bot manager" for role in guild.roles),
    #         "detailed_info": {
    #             "text_channels": len([c for c in guild.channels if isinstance(c, discord.TextChannel)]),
    #             "voice_channels": len([c for c in guild.channels if isinstance(c, discord.VoiceChannel)]),
    #             "categories": len([c for c in guild.channels if isinstance(c, discord.CategoryChannel)]),
    #             "emoji_count": len(guild.emojis),
    #             "sticker_count": len(guild.stickers),
    #             "system_channel": str(guild.system_channel.id) if guild.system_channel else None,
    #             "mfa_level": guild.mfa_level,
    #             "explicit_content_filter": str(guild.explicit_content_filter)
    #         }
    #     }
        
    #     try:
    #         for ph in posthog_instances:
    #             ph.group_identify(
    #                 "guild",
    #                 str(guild.id),
    #                 selfhost_properties
    #             )
    #             ph.capture(
    #                 distinct_id=str(self.bot.user.id),
    #                 event="guild_join",
    #                 properties=selfhost_properties
    #             )
    #             ph.flush()
    #     except Exception as e:
    #         log.error(f"Failed to capture guild join: {e}")

    # @Cog.listener("on_interaction")
    # async def on_interaction_analytics(self, interaction: discord.Interaction):
    #     if not interaction.type:
    #         return

    #     properties = {
    #         "interaction_type": str(interaction.type),
    #         "command_name": None,
    #         "command_type": None,
    #         "app_permissions": str(interaction.app_permissions) if interaction.app_permissions else None,
    #         "locale": str(interaction.locale),
    #         "response_time": round(interaction.client.latency * 1000, 2),
    #         "channel_type": str(interaction.channel.type) if interaction.channel else "Unknown",
    #         "is_dm": isinstance(interaction.channel, discord.DMChannel),
    #         "bot_in_guild": bool(interaction.guild and interaction.guild.me) if interaction.guild else False
    #     }

    #     if interaction.command:
    #         properties.update({
    #             "command_name": interaction.command.qualified_name if hasattr(interaction.command, 'qualified_name') else interaction.command.name,
    #             "command_type": (
    #                 str(interaction.command.type) 
    #                 if hasattr(interaction.command, 'type') 
    #                 else "hybrid" if isinstance(interaction.command, HybridCommand)
    #                 else "app" if isinstance(interaction.command, AppCommand)
    #                 else None
    #             ),
    #             "command_category": getattr(interaction.command, 'cog_name', None)
    #         })

    #     if interaction.guild:
    #         properties.update({
    #             "guild_id": str(interaction.guild.id),
    #             "guild_member_count": interaction.guild.member_count,
    #             "guild_has_bot": interaction.guild.me is not None,
    #             "user_is_owner": interaction.user.id == interaction.guild.owner_id,
    #             "user_permissions": [perm[0] for perm in interaction.permissions if perm[1]] if hasattr(interaction, 'permissions') else []
    #         })

    #     properties.update({
    #         "user_id": str(interaction.user.id),
    #         "user_bot": interaction.user.bot,
    #         "user_system": interaction.user.system,
    #         "user_created_at": interaction.user.created_at.isoformat()
    #     })

    #     properties.update({
    #         "cpu_percent": psutil.cpu_percent(),
    #         "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
    #         "thread_count": psutil.Process().num_threads()
    #     })

    #     try:
    #         for ph in posthog_instances:
    #             ph.capture(
    #                 str(interaction.user.id),
    #                 event="interaction",
    #                 properties=properties,
    #                 groups={"guild": str(interaction.guild.id)} if interaction.guild else None
    #             )
    #     except Exception as e:
    #         log.error(f"Failed to capture interaction analytics: {e}")

    # @Cog.listener("on_guild_remove")
    # async def on_guild_leave(self, guild: discord.Guild):
    #     """Track when the bot leaves a guild and clean up data"""
        
    #     properties = {
    #         "guild_id": str(guild.id),
    #         "guild_name": guild.name,
    #         "member_count": guild.member_count,
    #         "duration_days": (datetime.now(timezone.utc) - guild.me.joined_at).days if guild.me.joined_at else None,
    #         "reason": "unknown" 
    #     }
        
    #     for ph in posthog_instances:
    #         ph.capture(
    #             str(self.bot.user.id),
    #             event="guild_leave",
    #             properties=properties
    #         )
            
    #         ph.group_identify(
    #             "guild",
    #             str(guild.id),
    #             {
    #                 "deleted": True,
    #                 "deleted_at": datetime.now(timezone.utc).isoformat()
    #             }
    #         )
        
    #     log.info(f"Left guild: {guild.name} ({guild.id}) with {guild.member_count} members")

    @Cog.listener("on_application_command")
    async def track_slash_command(self, interaction: discord.Interaction):
        if not interaction.command:
            return
        
        properties = {
            "command_name": interaction.command.name,
            "command_type": str(interaction.command.type),
            "guild_id": str(interaction.guild_id) if interaction.guild else None,
            "channel_type": str(interaction.channel.type) if interaction.channel else None,
            "user_id": str(interaction.user.id),
            "app_permissions": str(interaction.app_permissions) if interaction.app_permissions else None,
            "response_time": round(interaction.client.latency * 1000, 2),
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
                "thread_count": psutil.Process().num_threads()
            }
        }
        
        for ph in posthog_instances:
            ph.capture(
                str(interaction.user.id),
                "slash_command_used",
                properties
            )
            ph.flush()

    # @Cog.listener("on_error")
    # async def track_error(self, event: str, *args, **kwargs):
    #     properties = {
    #         "event_name": event,
    #         "error_args": str(args),
    #         "error_kwargs": str(kwargs),
    #         "system_metrics": {
    #             "cpu_percent": psutil.cpu_percent(),
    #             "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
    #             "thread_count": psutil.Process().num_threads(),
    #             "python_version": sys.version,
    #             "platform": platform.platform()
    #         }
    #     }
        
    #     for ph in posthog_instances:
    #         ph.capture(
    #             "system",
    #             "error_occurred",
    #             properties
    #         )
    #         ph.flush()
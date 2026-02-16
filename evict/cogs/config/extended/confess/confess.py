import discord
import datetime

from discord import Embed, TextChannel, app_commands
from discord.ext.commands import has_permissions, hybrid_group

from core.client.context import Context

from .modals import confessModal

from asyncpg import UniqueViolationError
from managers.paginator import Paginator
from tools import CompositeMetaClass, MixinMeta


class Confess(MixinMeta, metaclass=CompositeMetaClass):
    """
    Anoymous confession commands.
    """

    @hybrid_group(invoke_without_command=True)
    async def confessions(self, ctx: Context):
        """
        Configure anonymous confessions.
        """
        return await ctx.send_help(ctx.command)

    @confessions.command(
        name="mute",
        example="34",
    )
    @has_permissions(manage_messages=True)
    async def confessions_mute(self, ctx: Context, *, number: int):
        """
        Mute a member that send a specific confession.
        """
        check = await self.bot.db.fetchrow(
            """
            SELECT channel_id 
            FROM confess 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id)
        )
        
        if check is None:
            return await ctx.warn("Confessions aren't **enabled** in this server!")

        re = await self.bot.db.fetchrow(
            """
            SELECT * FROM confess_members 
            WHERE guild_id = $1 AND 
            confession = $2
            """,
            ctx.guild.id,
            number,
        )
        
        if re is None:
            return await ctx.warn("**Couldn't** find that confession!")

        member_id = re["user_id"]

        r = await self.bot.db.fetchrow(
            """
            SELECT * FROM confess_mute 
            WHERE guild_id = $1 
            AND user_id = $2
            """,
            ctx.guild.id,
            member_id,
        )
        
        if r:
            return await ctx.warn("This **member** is **already** confession muted!")

        await self.bot.db.execute(
            """
            INSERT INTO confess_mute 
            VALUES ($1,$2)
            """, 
            ctx.guild.id, 
            member_id
        )
        
        return await ctx.approve(f"*Muted** the author of confession #{number}.")

    @confessions.command(
        name="unmute",
        example="34",
    )
    @has_permissions(manage_messages=True)
    async def confessions_unmute(self, ctx: Context, *, number: str):
        """
        Unmute a member that send a specific confession.
        """
        check = await self.bot.db.fetchrow(
            """
            SELECT channel_id 
            FROM confess 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id)
        )
        
        if check is None:
            return await ctx.warn("Confessions aren't **enabled** in this server!")

        if number == "all":
            await ctx.prompt("Are you sure you want to unmute **everyone** in confession mute?")
            await self.bot.db.execute(
                """
                DELETE FROM confess_mute 
                WHERE guild_id = $1
                """, 
                ctx.guild.id
            )
            
            return await ctx.approve("Unmuted **everyone** in confession mute!")

        num = int(number)
        
        re = await self.bot.db.fetchrow(
            """
            SELECT * FROM confess_members 
            WHERE guild_id = $1 
            AND confession = $2
            """,
            ctx.guild.id,
            num,
        )

        if re is None:
            return await ctx.warn("**Couldn't** find that confession!")
        
        member_id = re["user_id"]

        r = await self.bot.db.fetchrow(
            """
            SELECT * FROM confess_mute 
            WHERE guild_id = $1 
            AND user_id = $2
            """,
            ctx.guild.id,
            member_id,
        )
        
        if not r:
            return await ctx.warn("This **member** is **not** confession muted!")

        await self.bot.db.execute(
            """
            DELETE FROM confess_mute 
            WHERE guild_id = $1 
            AND user_id = $2
            """,
            ctx.guild.id,
            member_id,
        )
        
        return await ctx.approve(f"**Unmuted** the author of confession #{number}!")

    @confessions.command(
        name="add",
        example="#confessions",
    )
    @has_permissions(manage_guild=True)
    async def confessions_add(self, ctx: Context, *, channel: TextChannel):
        """
        Set the confession channel.
        """
        check = await self.bot.db.fetchrow(
            """
            SELECT * FROM confess 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id)
        )
        
        if check is not None:
            await self.bot.db.execute(
                """
                UPDATE confess 
                SET channel_id = $1 
                WHERE guild_id = $2
                """,
                channel.id,
                ctx.guild.id,
            )

        elif check is None:
            await self.bot.db.execute(
                """
                INSERT INTO confess 
                VALUES ($1,$2,$3)
                """, 
                ctx.guild.id, 
                channel.id, 
                0
            )
        
        return await ctx.approve(f"Set the confessions channel to {channel.mention}!")

    @confessions.command(
        name="remove",
    )
    @has_permissions(manage_guild=True)
    async def confessions_remove(self, ctx: Context):
        """
        Remove confession channel.
        """
        check = await self.bot.db.fetchrow(
            """
            SELECT channel_id 
            FROM confess 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id))

        if check is None:
            return await ctx.warn("Confessions aren't **enabled** in this server!")

        await self.bot.db.execute(
            """
            DELETE FROM confess 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id))
        
        await self.bot.db.execute(
            """
            DELETE FROM confess_members 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id))
        
        await self.bot.db.execute(
            """
            DELETE FROM confess_mute 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id))
        
        return await ctx.approve("**Disabled** confessions for this server!")

    @confessions.command(name="channel")
    async def confessions_channel(self, ctx: Context):
        """
        Get the confessions channel.
        """
        check = await self.bot.db.fetchrow(
            """
            SELECT * FROM confess 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id))

        if check is not None:

            channel = ctx.guild.get_channel(check["channel_id"])
            embed = Embed(description=f"Confession Channel: {channel.mention}\nConfessions Sent: **{check['confession']}**",)

            return await ctx.send(embed=embed)

        return await ctx.warn("Confessions aren't **enabled** in this server!")

    @app_commands.command(name="confess")
    async def confess(self, ctx: discord.Interaction):
        """
        Confess anonymously in the server.
        """
        re = await ctx.client.db.fetchrow(
            """
            SELECT * FROM confess_mute 
            WHERE guild_id = $1 
            AND user_id = $2
            """,
            ctx.guild.id,
            ctx.user.id,
        )

        if re:
            await ctx.warn(
                ctx,
                "You are **muted** from sending confessions in this server!",
                ephemeral=True,
            )

        check = await ctx.client.db.fetchrow(
            """
            SELECT channel_id 
            FROM confess 
            WHERE guild_id = {}
            """
            .format
            (ctx.guild.id)
        )

        if check:
            return await ctx.response.send_modal(confessModal())

        return await ctx.warn("Confessions **aren't** enabled in this server!")

    @confessions.group(name="blacklist", invoke_without_command=True)
    @has_permissions(manage_guild=True)
    async def confessions_blacklist(self, ctx: Context):
        """
        Manage confession blacklisted words.
        """
        return await ctx.send_help(ctx.command)

    @confessions_blacklist.command(name="add", example="badword")
    @has_permissions(manage_guild=True)
    async def blacklist_add(self, ctx: Context, *, word: str):
        """
        Add a word to the confession blacklist.
        """
        try:
            await self.bot.db.execute(
                """
                INSERT INTO confess_blacklist 
                (guild_id, word)
                VALUES ($1, $2)
                """,
                ctx.guild.id, word.lower()
            )
            
            return await ctx.approve(f"Added `{word}` to the confession blacklist")
        
        except UniqueViolationError:
            return await ctx.warn(f"`{word}` is already blacklisted")

    @confessions_blacklist.command(name="remove", example="badword")
    @has_permissions(manage_guild=True)
    async def blacklist_remove(self, ctx: Context, *, word: str):
        """
        Remove a word from the confession blacklist.
        """
        result = await self.bot.db.execute(
            """
            DELETE FROM confess_blacklist 
            WHERE guild_id = $1 
            AND word = $2
            """,
            ctx.guild.id, word.lower()
        )
        
        if result == "DELETE 0":
            return await ctx.warn(f"`{word}` is not blacklisted!")
        return await ctx.approve(f"Removed `{word}` from the confession blacklist")

    @confessions_blacklist.command(name="list")
    @has_permissions(manage_guild=True)
    async def blacklist_list(self, ctx: Context):
        """
        List all blacklisted words.
        """
        words = await self.bot.db.fetch(
            """
            SELECT word FROM confess_blacklist 
            WHERE guild_id = $1
            ORDER BY word
            """,
            ctx.guild.id
        )
        
        if not words:
            return await ctx.warn("No words are blacklisted")

        entries = [f"`{record['word']}`" for record in words]
        
        paginator = Paginator(
            ctx,
            entries=entries,
            embed=Embed(title="Confession Blacklisted Words"),
            per_page=20
        )
        return await paginator.start()

    @confessions_blacklist.command(name="clear")
    @has_permissions(manage_guild=True)
    async def blacklist_clear(self, ctx: Context):
        """
        Clear all blacklisted words.
        """
        await self.bot.db.execute(
            """
            DELETE FROM confess_blacklist 
            WHERE guild_id = $1
            """,
            ctx.guild.id
        )
        return await ctx.approve("Cleared all blacklisted words")

    @confessions.command(name="report", example="34")
    async def confessions_report(self, ctx: Context, confession_id: str):
        """
        Report a confession or reply.
        """
        report_channel = self.bot.get_channel(1333340804067889235)
        if not report_channel:
            return await ctx.warn("Report channel not found. Please contact staff.")

        try:
            if len(confession_id) > 10:
                message_id = int(confession_id)
                try:
                    check = await self.bot.db.fetchrow(
                        """
                        SELECT channel_id 
                        FROM confess 
                        WHERE guild_id = $1
                        """,
                        ctx.guild.id
                    )
                    if not check:
                        return await ctx.warn("Confessions **aren't** enabled in this server!")
                        
                    confess_channel = ctx.guild.get_channel(check['channel_id'])
                    if not confess_channel:
                        return await ctx.warn("Confession channel not found!")

                    message = None
                    for thread in confess_channel.threads:
                        try:
                            message = await thread.fetch_message(message_id)
                            if message:
                                break
                        except (discord.NotFound, discord.HTTPException):
                            continue
                    
                    if not message:
                        async for thread in confess_channel.archived_threads():
                            try:
                                message = await thread.fetch_message(message_id)
                                if message:
                                    break
                            except (discord.NotFound, discord.HTTPException):
                                continue
                    
                    if not message or not message.author.bot or not message.embeds:
                        return await ctx.warn("Reply not found.")

                    reply_author = await self.bot.db.fetchrow(
                        """
                        SELECT user_id 
                        FROM confess_replies 
                        WHERE message_id = $1
                        """,
                        message_id
                    )
                    
                    report_embed = Embed(
                        title="Confession Reply Report",
                        description=f"**Reply Content:**\n{message.embeds[0].description}",
                        timestamp=datetime.datetime.now(),
                        color=discord.Color.red()
                    )
                    report_embed.add_field(name="Reporter", value=f"{ctx.author} ({ctx.author.id})")
                    report_embed.add_field(name="Server", value=f"{ctx.guild.name} ({ctx.guild.id})")
                    report_embed.add_field(name="Channel", value=f"{confess_channel.mention} ({confess_channel.id})")
                    report_embed.add_field(name="Message ID", value=message_id)
                    if reply_author:
                        report_embed.add_field(name="Reply Author ID", value=reply_author['user_id'])
                    
                    await report_channel.send(embed=report_embed)
                    return await ctx.approve("Report sent successfully.")
                except discord.NotFound:
                    return await ctx.warn("Reply not found.")
            
            else:
                confession_num = int(confession_id)
                confession_data = await self.bot.db.fetchrow(
                    """
                    SELECT cm.user_id, cm.guild_id, c.channel_id, c.confession 
                    FROM confess_members cm
                    JOIN confess c ON c.guild_id = cm.guild_id
                    WHERE cm.guild_id = $1 AND cm.confession = $2
                    """,
                    ctx.guild.id, confession_num
                )
                
                if not confession_data:
                    return await ctx.warn("Confession not found.")

                channel = ctx.guild.get_channel(confession_data['channel_id'])
                if not channel:
                    return await ctx.warn("Confession channel not found.")

                async for message in channel.history(limit=100):
                    if (message.author.bot and message.embeds 
                        and message.embeds[0].author.name == f"anonymous confession #{confession_num}"):
                        report_embed = Embed(
                            title="Confession Report",
                            description=f"**Confession #{confession_num}:**\n{message.embeds[0].description}",
                            timestamp=datetime.datetime.now(),
                            color=discord.Color.red()
                        )
                        report_embed.add_field(name="Reporter", value=f"{ctx.author} ({ctx.author.id})")
                        report_embed.add_field(name="Server", value=f"{ctx.guild.name} ({ctx.guild.id})")
                        report_embed.add_field(name="Confession Author ID", value=confession_data['user_id'])
                        
                        await report_channel.send(embed=report_embed)
                        return await ctx.approve("Report sent successfully.")

                return await ctx.warn("Couldn't find the confession message.")

        except ValueError:
            return await ctx.warn("Invalid confession ID or message ID.")

    @confessions.group(name="emojis", invoke_without_command=True)
    @has_permissions(manage_guild=True)
    async def confessions_emojis(self, ctx: Context):
        """
        Manage confession reaction emojis.
        """
        return await ctx.send_help(ctx.command)

    @confessions_emojis.command(name="set", example="👍 👎")
    @has_permissions(manage_guild=True)
    async def emojis_set(self, ctx: Context, upvote: str, downvote: str):
        """
        Set custom reaction emojis for confessions.
        Use 'none' to disable reactions.
        """
        if upvote.lower() != "none":
            try:
                await ctx.message.add_reaction(upvote)
                await ctx.message.remove_reaction(upvote, ctx.me)
            except discord.HTTPException:
                return await ctx.warn(f"Invalid upvote emoji: {upvote}")

        if downvote.lower() != "none":
            try:
                await ctx.message.add_reaction(downvote)
                await ctx.message.remove_reaction(downvote, ctx.me)
            except discord.HTTPException:
                return await ctx.warn(f"Invalid downvote emoji: {downvote}")

        await self.bot.db.execute(
            """
            UPDATE confess 
            SET upvote = $1, downvote = $2 
            WHERE guild_id = $3
            """,
            None if upvote.lower() == "none" else upvote,
            None if downvote.lower() == "none" else downvote,
            ctx.guild.id
        )

        if upvote.lower() == "none" and downvote.lower() == "none":
            return await ctx.approve("Disabled confession reactions")
        return await ctx.approve(f"Set confession reactions to {upvote} and {downvote}")

    @confessions_emojis.command(name="reset")
    @has_permissions(manage_guild=True)
    async def emojis_reset(self, ctx: Context):
        """
        Reset confession reaction emojis to default (👍 👎).
        """
        await self.bot.db.execute(
            """
            UPDATE confess 
            SET upvote = $1, downvote = $2 
            WHERE guild_id = $3
            """,
            "👍", "👎", ctx.guild.id
        )
        return await ctx.approve("Reset confession reactions to default (👍 👎)")

    @confessions_emojis.command(name="view")
    @has_permissions(manage_guild=True)
    async def emojis_view(self, ctx: Context):
        """
        View current confession reaction emojis.
        """
        data = await self.bot.db.fetchrow(
            """
            SELECT upvote, downvote 
            FROM confess 
            WHERE guild_id = $1
            """,
            ctx.guild.id
        )
        
        if not data:
            return await ctx.warn("Confessions are not set up in this server!")
            
        if not data['upvote'] and not data['downvote']:
            return await ctx.warn("Confession reactions are disabled")
            
        return await ctx.neutral(f"Current confession reactions: {data['upvote']} {data['downvote']}")

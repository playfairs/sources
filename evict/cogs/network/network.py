# POSTHOG -> HOG.EVICT.BOT
# THIS HAS BEEN DEPRECATED -> TOO MANY REQUESTS WHICH SLOWS DOWN THE BOT

import os
import json
import logging
import config
import asyncio
import time
import discord
import aiohttp
import stripe
import orjson
from discord import Embed
import hashlib
import hmac
from random import choice
import functools
import traceback
from typing import Any, Dict, List, Optional
import re
import uuid
from urllib.parse import urlencode
import dns.resolver
import secrets
import random
import string
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from main import Evict

from aiohttp import web, WSMsgType
from aiohttp.abc import AbstractAccessLogger
from aiohttp_cors import setup as cors_setup, ResourceOptions
from aiohttp.web import BaseRequest, Request, Response, StreamResponse

from discord import ActivityType, Spotify, Streaming, Status
from itertools import groupby

from discord.ext.commands import Cog, command, group

from functools import wraps
from typing import Callable, Optional
from cashews import cache
from pomice import LoopMode
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel  
from collections import defaultdict
from discord import Status
from discord.ext.commands import Group, FlagConverter

import config
from posthog import Posthog

CONFIG = {
    "github_allowed_repos": [
        "EvictServices/evict.new", 
    ],
    "token": config.DISCORD.TOKEN, 
    "updates_channel_id": 1319095893831581697 
}

class Owner(BaseModel):
    name: Optional[str] = None
    email: Optional[Any] = None
    login: Optional[str] = None
    id: Optional[int] = None
    node_id: Optional[str] = None
    avatar_url: Optional[str] = None
    gravatar_id: Optional[str] = None
    url: Optional[str] = None
    html_url: Optional[str] = None
    followers_url: Optional[str] = None
    following_url: Optional[str] = None
    gists_url: Optional[str] = None
    starred_url: Optional[str] = None
    subscriptions_url: Optional[str] = None
    organizations_url: Optional[str] = None
    repos_url: Optional[str] = None
    events_url: Optional[str] = None
    received_events_url: Optional[str] = None
    type: Optional[str] = None
    user_view_type: Optional[str] = None
    site_admin: Optional[bool] = None


class License(BaseModel):
    key: Optional[str] = None
    name: Optional[str] = None
    spdx_id: Optional[str] = None
    url: Optional[str] = None
    node_id: Optional[str] = None


class Repository(BaseModel):
    id: Optional[int] = None
    node_id: Optional[str] = None
    name: Optional[str] = None
    full_name: Optional[str] = None
    private: Optional[bool] = None
    owner: Optional[Owner] = None
    html_url: Optional[str] = None
    description: Optional[Any] = None
    fork: Optional[bool] = None
    url: Optional[str] = None
    forks_url: Optional[str] = None
    keys_url: Optional[str] = None
    collaborators_url: Optional[str] = None
    teams_url: Optional[str] = None
    hooks_url: Optional[str] = None
    issue_events_url: Optional[str] = None
    events_url: Optional[str] = None
    assignees_url: Optional[str] = None
    branches_url: Optional[str] = None
    tags_url: Optional[str] = None
    blobs_url: Optional[str] = None
    git_tags_url: Optional[str] = None
    git_refs_url: Optional[str] = None
    trees_url: Optional[str] = None
    statuses_url: Optional[str] = None
    languages_url: Optional[str] = None
    stargazers_url: Optional[str] = None
    contributors_url: Optional[str] = None
    subscribers_url: Optional[str] = None
    subscription_url: Optional[str] = None
    commits_url: Optional[str] = None
    git_commits_url: Optional[str] = None
    comments_url: Optional[str] = None
    issue_comment_url: Optional[str] = None
    contents_url: Optional[str] = None
    compare_url: Optional[str] = None
    merges_url: Optional[str] = None
    archive_url: Optional[str] = None
    downloads_url: Optional[str] = None
    issues_url: Optional[str] = None
    pulls_url: Optional[str] = None
    milestones_url: Optional[str] = None
    notifications_url: Optional[str] = None
    labels_url: Optional[str] = None
    releases_url: Optional[str] = None
    deployments_url: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[int] = None
    git_url: Optional[str] = None
    ssh_url: Optional[str] = None
    clone_url: Optional[str] = None
    svn_url: Optional[str] = None
    homepage: Optional[Any] = None
    size: Optional[int] = None
    stargazers_count: Optional[int] = None
    watchers_count: Optional[int] = None
    language: Optional[str] = None
    has_issues: Optional[bool] = None
    has_projects: Optional[bool] = None
    has_downloads: Optional[bool] = None
    has_wiki: Optional[bool] = None
    has_pages: Optional[bool] = None
    has_discussions: Optional[bool] = None
    forks_count: Optional[int] = None
    mirror_url: Optional[Any] = None
    archived: Optional[bool] = None
    disabled: Optional[bool] = None
    open_issues_count: Optional[int] = None
    license: Optional[License] = None
    allow_forking: Optional[bool] = None
    is_template: Optional[bool] = None
    web_commit_signoff_required: Optional[bool] = None
    topics: Optional[List] = None
    visibility: Optional[str] = None
    forks: Optional[int] = None
    open_issues: Optional[int] = None
    watchers: Optional[int] = None
    default_branch: Optional[str] = None
    stargazers: Optional[int] = None
    master_branch: Optional[str] = None
    organization: Optional[str] = None
    custom_properties: Optional[Dict[str, Any]] = None


class Pusher(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


class Organization(BaseModel):
    login: Optional[str] = None
    id: Optional[int] = None
    node_id: Optional[str] = None
    url: Optional[str] = None
    repos_url: Optional[str] = None
    events_url: Optional[str] = None
    hooks_url: Optional[str] = None
    issues_url: Optional[str] = None
    members_url: Optional[str] = None
    public_members_url: Optional[str] = None
    avatar_url: Optional[str] = None
    description: Optional[str] = None


class Sender(BaseModel):
    login: Optional[str] = None
    id: Optional[int] = None
    node_id: Optional[str] = None
    avatar_url: Optional[str] = None
    gravatar_id: Optional[str] = None
    url: Optional[str] = None
    html_url: Optional[str] = None
    followers_url: Optional[str] = None
    following_url: Optional[str] = None
    gists_url: Optional[str] = None
    starred_url: Optional[str] = None
    subscriptions_url: Optional[str] = None
    organizations_url: Optional[str] = None
    repos_url: Optional[str] = None
    events_url: Optional[str] = None
    received_events_url: Optional[str] = None
    type: Optional[str] = None
    user_view_type: Optional[str] = None
    site_admin: Optional[bool] = None


class Author(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None


class Committer(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None


class Commit(BaseModel):
    id: Optional[str] = None
    tree_id: Optional[str] = None
    distinct: Optional[bool] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    url: Optional[str] = None
    author: Optional[Author] = None
    committer: Optional[Committer] = None
    added: Optional[List] = None
    removed: Optional[List] = None
    modified: Optional[List[str]] = None


class HeadCommit(BaseModel):
    id: Optional[str] = None
    tree_id: Optional[str] = None
    distinct: Optional[bool] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    url: Optional[str] = None
    author: Optional[Author] = None
    committer: Optional[Committer] = None
    added: Optional[List] = None
    removed: Optional[List] = None
    modified: Optional[List[str]] = None


class GithubPushEvent(BaseModel):
    ref: Optional[str] = None
    before: Optional[str] = None
    after: Optional[str] = None
    repository: Optional[Repository] = None
    pusher: Optional[Pusher] = None
    organization: Optional[Organization] = None
    sender: Optional[Sender] = None
    created: Optional[bool] = None
    deleted: Optional[bool] = None
    forced: Optional[bool] = None
    base_ref: Optional[Any] = None
    compare: Optional[str] = None
    commits: Optional[List[Commit]] = None
    head_commit: Optional[HeadCommit] = None

    @property
    def to_embed(self) -> Embed:
        if not self.head_commit:
            return None
            
        added_count = len(self.head_commit.added or [])
        deleted_count = len(self.head_commit.removed or [])
        modified_count = len(self.head_commit.modified or [])

        # Construct each part of the message based on conditions
        added_message = (
            f"+ Added {added_count} {'files' if added_count > 1 else 'file'}"
            if added_count > 0
            else ""
        )
        deleted_message = (
            f"- Deleted {deleted_count} {'files' if deleted_count > 1 else 'file'}"
            if deleted_count > 0
            else ""
        )
        modified_message = (
            f"! Modified {modified_count} {'files' if modified_count > 1 else 'file'}"
            if modified_count > 0
            else ""
        )

        # Combine non-empty parts with newlines
        change_message = "\n".join(
            filter(None, [added_message, deleted_message, modified_message])
        )

        branch = self.ref.split('/')[-1]
        commit_count = len(self.commits or [])
        
        description = (
            f">>> There has been **{commit_count}** {'commit' if commit_count == 1 else 'commits'} "
            f"to [`{self.repository.full_name}`]({self.repository.html_url}/tree/{branch})\n"
            f"```diff\n{change_message}\n```"
        )

        embed = Embed(
            title=f"New {'Commit' if commit_count == 1 else 'Commits'} to {self.repository.name} ({branch})",
            url=self.compare,  
            description=description,
            color=0x2ea043  
        )

        valid_commit = False
        for commit in (self.commits or []):
            if commit.message and len(commit.message.strip()) >= 5:
                valid_commit = True
                commit_url = f"{self.repository.html_url}/commit/{commit.id}"
                embed.add_field(
                    name=f"{commit.id[:7]}",
                    value=f"[View Commit]({commit_url})\n```fix\n{commit.message.strip()}\n```",
                    inline=False,
                )

        if not valid_commit:
            return None

        if self.sender:
            embed.set_author(
                name=str(self.sender.login),
                icon_url=str(self.sender.avatar_url),
                url=str(self.sender.html_url)
            )
            
        embed.set_footer(
            text=f"📁 {self.repository.size or 0}KB | 📝 {self.repository.open_issues_count or 0} issues | 👥 {self.repository.watchers_count or 0} watchers",
            icon_url="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
        )
            
        embed.timestamp = datetime.now()
        return embed

    async def send_message(self):
        if not (embed := self.to_embed):
            return
        log.info(f"Received embed: {embed}")
        channel_id = CONFIG["updates_channel_id"]
        for _ in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"https://discord.com/api/v10/channels/{channel_id}/messages",
                        headers={
                            "Authorization": f"Bot {CONFIG['token']}",
                            "Content-Type": "application/json"
                        },
                        json={"embeds": [embed.to_dict()]}
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        
                        if response.status != 429:
                            log.error(f"Failed to send message: {response.status}")
                            
                await asyncio.sleep(1)
                        
            except Exception as e:
                log.error(f"Error sending message: {e}")
                await asyncio.sleep(1)

    async def send_message(self):
        if not (embed := self.to_embed):
            return

        channel_id = CONFIG["updates_channel_id"]
        token = CONFIG["token"]

        for _ in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"https://discord.com/api/v10/channels/{channel_id}/messages",
                        headers={
                            "Authorization": f"Bot {token}",
                            "Content-Type": "application/json"
                        },
                        json={"embeds": [embed.to_dict()]}
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        
                        if response.status != 429: 
                            log.error(f"Failed to send message: {response.status}")
                            
                await asyncio.sleep(1)  
                        
            except Exception as e:
                log.error(f"Error sending message: {e}")
                await asyncio.sleep(1)

        return None


class Module(BaseModel):
    threshold: int
    duration: int
    
    def __init__(self, **data):
        super().__init__(**data)
        self.last_trigger = 0
        self.count = 0


class Settings(BaseModel):
    bot: bool = False
    ban: Optional[Module] = None
    kick: Optional[Module] = None
    role: Optional[Module] = None
    channel: Optional[Module] = None
    webhook: Optional[Module] = None
    emoji: Optional[Module] = None

    @classmethod
    async def fetch(cls, bot, guild):
        data = await bot.db.fetchrow(
            "SELECT * FROM antinuke WHERE guild_id = $1", guild.id
        )
        return cls(**data) if data else cls()

    @classmethod
    async def revalidate(cls, bot, guild):
        await bot.cache.delete(f"antinuke:{guild.id}")


cache.setup("mem://")
log = logging.getLogger(__name__)


class AccessLogger(AbstractAccessLogger):
    def log(
        self: "AccessLogger",
        request: BaseRequest,
        response: StreamResponse,
        time: float,
    ) -> None:
        self.logger.info(
            f"Request for {request.path!r} with status of {response.status!r}."
        )


def requires_auth(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(self: "Network", request: Request) -> Response:
        auth_header = request.headers.get("Authorization")
        if auth_header != "":
            return web.json_response({"error": "Unauthorized"}, status=401)
        return await func(self, request)

    return wrapper

def requires_not_auth(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(self: "Network", request: Request) -> Response:
        auth_header = request.headers.get("Authorization")
        if auth_header != "":
            return web.json_response({"error": "Unauthorized"}, status=401)
        return await func(self, request)

    return wrapper

def ratelimit(requests: int, window: int):
    def decorator(func):
        async def wrapper(self, request: Request, *args, **kwargs):
            global_key = "global_ratelimit"
            global_log_key = "global_ratelimit_log"
            
            ip = (
                request.headers.get("CF-Connecting-IP")
                or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or request.remote
            )
            
            global_current = await self.bot.redis.get(global_key)
            if global_current and int(global_current) >= 10000:
                if not await self.bot.redis.exists(global_log_key):
                    log.info(f"Global rate limit exceeded across all endpoints")
                    await self.bot.redis.set(global_log_key, "1", ex=180)
                return web.json_response({"error": "Global rate limit exceeded"}, status=429)
            
            key = f"ratelimit:{ip}:{func.__name__}"
            log_key = f"ratelimit_log:{ip}:{func.__name__}"
            abuse_key = f"abuse:{ip}"
            
            abuse_count = await self.bot.redis.incr(abuse_key)
            if abuse_count == 1:
                await self.bot.redis.expire(abuse_key, 3600)
            
            current = await self.bot.redis.get(key)
            if current and int(current) >= requests:
                if not await self.bot.redis.exists(log_key):
                    abuse_log = {
                        "ip": ip,
                        "endpoint": func.__name__,
                        "user_agent": request.headers.get("User-Agent", "Unknown"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "abuse_count": abuse_count,
                        "path": request.path,
                        "method": request.method,
                        "headers": dict(request.headers)
                    }
                    
                    if abuse_count >= 50:  
                        try:
                            try:
                                with open("abuse_log.json", "r") as f:
                                    content = f.read()
                                    data = json.loads(content) if content else []
                            except (FileNotFoundError, json.JSONDecodeError):
                                data = []
                            
                            data.append(abuse_log)
                            
                            with open("abuse_log.json", "w") as f:
                                json.dump(data, f, indent=4)
                        except Exception as e:
                            log.error(f"Failed to log abuse: {e}")
                    
                    log.warning(f"Rate limit abuse from {ip} on {func.__name__} (Count: {abuse_count})")
                    await self.bot.redis.set(log_key, "1", ex=window)
                return web.json_response({"error": "Rate limit exceeded"}, status=429)
                
            pipe = self.bot.redis.pipeline()
            pipe.incr(key)
            if not current:
                pipe.expire(key, window)
            await pipe.execute()
            
            response = await func(self, request, *args, **kwargs)
            
            if response.status < 400:
                pipe = self.bot.redis.pipeline()
                pipe.incr(global_key)
                if not global_current:
                    pipe.expire(global_key, 60)
                await pipe.execute()
            
            return response
        return wrapper
    return decorator

def requires_special_auth(f):
    @functools.wraps(f)
    async def wrapped(self, request: Request, *args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return web.json_response({"error": "Unauthorized"}, status=401)
        
        token = auth_header.split(" ")[1]
        user_id = request.query.get("user_id")
        
        if not user_id or not await self.verify_token(token, int(user_id)):
            return web.json_response({"error": "Invalid token"}, status=401)
            
        return await f(self, request, *args, **kwargs)
    return wrapped


def route(pattern: str, method: str | list[str] = "GET") -> Callable:
    def decorator(func: Callable) -> Callable: 
        @wraps(func)
        async def wrapper(self: "Network", request: Request) -> None:
            start_time = datetime.now(timezone.utc)
            allowed_methods = [method] if isinstance(method, str) else method.copy()
            
            if request.method not in allowed_methods:
                response = web.json_response(
                    {"error": f"Method {request.method} not allowed"}, 
                    status=405,
                    headers={"Allow": ", ".join(allowed_methods)}
                )
                # posthog_self.capture(
                #     "anonymous",
                #     "api_request",
                #     {
                #         "endpoint": pattern,
                #         "method": request.method,
                #         "status_code": 405,
                #         "error": "Method not allowed",
                #         "response_time": round((datetime.now(timezone.utc) - start_time).total_seconds() * 1000, 2),
                #         "user_agent": request.headers.get("User-Agent"),
                #         "ip": request.headers.get("X-Real-IP") or request.remote
                #     }
                # )
                return response
            
            try:
                response = await func(self, request)
                # posthog_self.capture(
                #     "anonymous",
                #     "api_request",
                #     {
                #         "endpoint": pattern,
                #         "method": request.method,
                #         "status_code": response.status,
                #         "response_time": round((datetime.now(timezone.utc) - start_time).total_seconds() * 1000, 2),
                #         "user_agent": request.headers.get("User-Agent"),
                #         "ip": request.headers.get("X-Real-IP") or request.remote
                #     }
                # )
                return response
            except Exception as e:
                log.error(f"Error in {pattern}: {e}")
                response = web.json_response(
                    {"error": "Internal server error"}, 
                    status=500
                )
                # posthog_self.capture(
                #     "anonymous",
                #     "api_request",
                #     {
                #         "endpoint": pattern,
                #         "method": request.method,
                #         "status_code": 500,
                #         "error": str(e),
                #         "response_time": round((datetime.now(timezone.utc) - start_time).total_seconds() * 1000, 2),
                #         "user_agent": request.headers.get("User-Agent"),
                #         "ip": request.headers.get("X-Real-IP") or request.remote
                #     }
                # )
                return response

        wrapper.methods = [method] if isinstance(method, str) else method.copy()
        wrapper.pattern = pattern
        return wrapper
    return decorator


class Network(Cog):
    def __init__(self, bot: Evict):
        self.bot: Evict = bot
        self.app = web.Application(
            client_max_size=1024**2, 
            middlewares=[]
        )
        self.runner = None 
        self.site = None  
        
        self.cors = cors_setup(
            self.app,
            defaults={
                "*": ResourceOptions(
                    allow_credentials=False,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods=["GET", "POST", "OPTIONS"],
                    max_age=3600
                )
            },
        )

        async def request_middleware(request, handler):
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]

            try:
                response = await handler(request)
                
                if response.status == 429:
                    ratelimit_key = f"ratelimit_log:{request.remote}:{request.path}"
                    if not await self.bot.redis.exists(ratelimit_key):
                        log.info(f"[{request_id}] Rate limited {request.method} {request.path} from {request.remote}")
                        await self.bot.redis.set(ratelimit_key, "1", ex=60)
                else:
                    log_key = f"weblogs:{request.path}:{request.remote}"
                    should_log = not await self.bot.redis.exists(log_key)
                    if should_log:
                        log.info(f"[{request_id}] {request.method} {request.path} from {request.remote}")
                        await self.bot.redis.set(log_key, "1", ex=60)
                
                return response
                    
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                log.error(
                    f"[{request_id}] Request timed out after {duration:.2f}s: "
                    f"{request.method} {request.path}"
                )
                return web.json_response(
                    {"error": "Request timed out"}, 
                    status=504
                )
                
            except Exception as e:
                duration = time.time() - start_time
                log.error(
                    f"[{request_id}] Error handling request after {duration:.2f}s: "
                    f"{request.method} {request.path}"
                )
                log.error(f"[{request_id}] Error details: {e}")
                log.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
                
                return web.json_response(
                    {"error": "Internal server error"},
                    status=500
                )

        self.app.middlewares.append(web.middleware(request_middleware))

        for module in dir(self):
            route = getattr(self, module)
            if not hasattr(route, "pattern"):
                continue
            
            resource = self.app.router.add_resource(route.pattern)
            
            methods = route.methods if isinstance(route.methods, list) else [route.methods]
            methods = [m for m in methods if m != "OPTIONS"]
            
            for method in methods:
                handler = resource.add_route(method, route)
                self.cors.add(handler)

        root_resource = self.app.router.add_get("/", self.root_handler)
        self.cors.add(root_resource)

        self.ws_connections = {}  # {guild_id: {auth_token: ws}}
        self.app.router.add_get("/ws/music/{guild_id}", self.music_websocket)

        self.failed_payment_notifications = defaultdict(list)

    def required_xp(self, level: int, multiplier: int = 1) -> int:
        """
        Calculate the required XP for a given level.
        """
        xp = sum((i * 100) + 75 for i in range(level))
        return int(xp * multiplier)

    async def root_handler(self, request):
        return web.json_response(
            {
            "commands": {len([cmd for cmd in self.bot.walk_commands() if cmd.cog_name != 'Jishaku' and cmd.cog_name != 'Owner'])},
            "latency": self.bot.latency * 1000,
            "cache": {
                "guilds": len(self.bot.guilds),
                "users": len([user for user in self.bot.users if not user.bot]),
            },
            }
        )

    async def cog_load(self) -> None:
        host = config.NETWORK.HOST
        port = config.NETWORK.PORT
        
        if hasattr(self, 'app'):
            await self.app.cleanup()
            
        self.app = web.Application(
            client_max_size=1024**2,  
            middlewares=[
                
            ]
        )
        self.cors = cors_setup(
            self.app,
            defaults={
                "*": ResourceOptions(
                    allow_credentials=False,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods=["GET", "POST", "OPTIONS"],
                    max_age=3600
                )
            },
        )
        
        for module in dir(self):
            route = getattr(self, module)
            if not hasattr(route, "pattern"):
                continue
            
            resource = self.app.router.add_resource(route.pattern)
            for method in route.methods:
                handler = resource.add_route(method, route)
                self.cors.add(handler)
                
        self.app.router.add_get("/music/{guild_id}", self.music_websocket)
        
        self.runner = web.AppRunner(
            self.app,
            access_log=log, 
            handle_signals=True,
            keepalive_timeout=75.0, 
            tcp_keepalive=True,
            shutdown_timeout=60.0
        )
        await self.runner.setup()
        
        self.site = web.TCPSite(
            self.runner, 
            host, 
            port,
            backlog=1024,
            reuse_address=True,
            reuse_port=True
        )
        await self.site.start()
        
        log.info(f"Started the internal API on {host}:{port}.")

    async def cog_unload(self) -> None:
        if self.site:
            await self.site.stop()
            log.info("Stopped the TCP site")
        
        if self.runner:
            await self.runner.cleanup()
            log.info("Cleaned up the runner")
        
        await self.app.shutdown()
        await self.app.cleanup()
        log.info("Gracefully shutdown the API")

    @route("/health")
    async def health(self, request: Request) -> Response:
        return web.json_response({
            "status": "ok",
            "timestamp": int(time.time()),
            "uptime": int(self.bot.uptime2)
        })

    @route("/commands")
    async def commands(self: "Network", request: Request) -> Response:
        """
        Export command information as JSON, including commands within groups.
        """
        def get_flags(param):
            if isinstance(param.annotation, type) and issubclass(param.annotation, FlagConverter):
                flags = param.annotation.get_flags()
                return {
                    "required": [
                        {
                            "name": name,
                            "description": flag.description
                        }
                        for name, flag in flags.items()
                        if not flag.default
                    ],
                    "optional": [
                        {
                            "name": name,
                            "description": flag.description
                        }
                        for name, flag in flags.items()
                        if flag.default
                    ]
                }
            return None
        
        def get_donator(command):
            if command.checks:
                for check in command.checks:
                    if check.__name__ == "predicate" and check.__qualname__.startswith("donator"):
                        return True
            return False

        def get_permissions(command):
            try:
                perms = [perm.lower().replace("n/a", "None").replace("_", " ") 
                        for perm in command.permissions]
                
                if "antinuke" in command.qualified_name.lower():
                    perms.extend(["antinuke admin", "guild owner"])
                    
                if len(perms) > 1:
                    perms = [p for p in perms if p.lower() not in ("none", "n/a")]
                    
                return perms
            except AttributeError:
                perms = []
                for check in command.checks if command.checks else []:
                    if hasattr(check, 'closure') and check.closure:
                        for cell in check.closure:
                            if hasattr(cell, 'cell_contents') and isinstance(cell.cell_contents, dict):
                                perms.extend(cell.cell_contents.keys())
                                
                if "antinuke" in command.qualified_name.lower():
                    perms.extend(["antinuke admin", "guild owner"])
                    
                perms = [perm.replace('_', ' ').title() for perm in perms] if perms else ["N/A"]
                
                if len(perms) > 1:
                    perms = [p for p in perms if p.lower() not in ("n/a", "none")]
                    
                return perms

        def format_parameters(command):
            def clean_type(annotation):
                if hasattr(annotation, '__name__'):
                    return annotation.__name__
                if str(annotation).startswith('<'): 
                    return str(annotation.__class__.__name__)
                if str(annotation).startswith('typing.Optional'):
                    return 'Optional[' + clean_type(annotation.__args__[0]) + ']'
                return str(annotation)

            return [
                {
                    "name": name,
                    "type": clean_type(param.annotation),
                    "default": None if param.default == param.empty else str(param.default),
                    "flags": get_flags(param),
                    "optional": param.default != param.empty
                }
                for name, param in command.clean_params.items()
            ]

        IGNORED_CATEGORIES = [
            "Jishaku",
            "Network",
            "API", 
            "Owner",
            "Status",
            "Listeners",
            "Hog"
        ]

        commands_info = []
        categories = sorted(list(set([
            cog.qualified_name for cog in self.bot.cogs.values() 
            if cog.qualified_name not in IGNORED_CATEGORIES
            and "cogs" in getattr(cog, "__module__", "")
        ])))

        for cog in self.bot.cogs.values():
            if cog.qualified_name in IGNORED_CATEGORIES:
                continue

            for command in cog.get_commands():
                if isinstance(command, Group):
                    commands_info.append({
                        "name": command.qualified_name,
                        "description": command.description or command.help or "No description",
                        "aliases": command.aliases,
                        "parameters": format_parameters(command),
                        "category": command.cog.qualified_name if command.cog else "No Category",
                        "permissions": get_permissions(command),
                        "donator": get_donator(command) 
                    })
                    
                    seen_commands = {command.qualified_name}
                    
                    for subcommand in command.walk_commands():
                        if subcommand.qualified_name not in seen_commands:
                            seen_commands.add(subcommand.qualified_name)
                            commands_info.append({
                                "name": subcommand.qualified_name, 
                                "description": subcommand.description or subcommand.help or "No description", 
                                "aliases": subcommand.aliases,
                                "parameters": format_parameters(subcommand),
                                "category": subcommand.cog.qualified_name if subcommand.cog else "No Category",
                                "permissions": get_permissions(subcommand),
                                "donator": get_donator(subcommand)
                            })
                else:
                    commands_info.append({
                        "name": command.qualified_name,
                        "description": command.description or command.help or "No description",
                        "aliases": command.aliases,
                        "parameters": format_parameters(command),
                        "category": command.cog.qualified_name if command.cog else "No Category", 
                        "permissions": get_permissions(command),
                        "donator": get_donator(command)
                    })

        return web.json_response({"categories": categories, "commands": commands_info})

    # @route("/status")
    # async def status(self, request: Request) -> Response:
    #     current_ts = int(time.time() * 1000)
        
        
    #     historical_data = await self.bot.db.fetch(
    #         """
    #         WITH dates AS (
    #             SELECT generate_series(
    #                 date_trunc('day', now() - interval '90 days'),
    #                 date_trunc('day', now()),
    #                 '1 day'::interval
    #             )::date as date
    #         ),
    #         daily_metrics AS (
    #             SELECT 
    #                 date(to_timestamp(timestamp / 1000)) as date,
    #                 COUNT(*) as update_count
    #             FROM status_metrics
    #             WHERE timestamp >= extract(epoch from (now() - interval '90 days')) * 1000
    #             GROUP BY date(to_timestamp(timestamp / 1000))
    #         )
    #         SELECT 
    #             d.date,
    #             h.incidents,
    #             CASE 
    #                 WHEN h.date IS NULL THEN 'no_data'
    #                 WHEN h.incidents IS NULL OR h.incidents = '[]' THEN 'no_incidents'
    #                 ELSE 'has_incidents'
    #             END as status,
    #             COALESCE(m.update_count, 0) as update_count
    #         FROM dates d
    #         LEFT JOIN status_history h ON d.date = h.date
    #         LEFT JOIN daily_metrics m ON d.date = m.date
    #         ORDER BY d.date ASC
    #         """
    #     )
        
    #     eight_hours_ago = current_ts - (8 * 60 * 60 * 1000)
    #     seven_days_ago = current_ts - (7 * 24 * 60 * 60 * 1000)
        
    #     metrics_history = await self.bot.db.fetch(
    #         """
    #         WITH hours AS (
    #             SELECT generate_series(
    #                 date_trunc('hour', to_timestamp($1::bigint / 1000)),
    #                 date_trunc('hour', to_timestamp($2::bigint / 1000)),
    #                 '1 hour'
    #             ) as hour
    #         ),
    #         hourly_metrics AS (
    #             SELECT 
    #                 date_trunc('hour', to_timestamp(timestamp / 1000)) as hour,
    #                 ROUND(AVG(cpu_usage)::numeric, 2) as avg_cpu,
    #                 ROUND(AVG(memory_usage)::numeric, 2) as avg_memory,
    #                 COUNT(*) as sample_count
    #             FROM status_metrics
    #             WHERE timestamp >= $1
    #             GROUP BY date_trunc('hour', to_timestamp(timestamp / 1000))
    #         )
    #         SELECT 
    #             EXTRACT(EPOCH FROM h.hour)::bigint * 1000 as timestamp,
    #             m.avg_cpu,
    #             m.avg_memory,
    #             m.sample_count
    #         FROM hours h
    #         LEFT JOIN hourly_metrics m ON h.hour = m.hour
    #         ORDER BY h.hour DESC
    #         """,
    #         eight_hours_ago,
    #         current_ts
    #     )

    #     seven_days_ago = current_ts - (7 * 24 * 60 * 60 * 1000)

    #     latency_history = await self.bot.db.fetch("""
    #     WITH hours AS (
    #         SELECT generate_series(
    #             date_trunc('hour', to_timestamp($1::bigint / 1000)),
    #             date_trunc('hour', to_timestamp($2::bigint / 1000)),
    #             '1 hour'
    #         ) as hour
    #     ),
    #     unnested AS (
    #         SELECT 
    #             jsonb_array_elements_text(sh.latency_metrics)::jsonb as metrics
    #         FROM status_history sh
    #         WHERE date >= CURRENT_DATE - interval '7 days'
    #     ),
    #     parsed AS (
    #         SELECT 
    #             date_trunc('hour', to_timestamp((jsonb_array_elements(metrics)->>'timestamp')::bigint / 1000)) as hour,
    #             (jsonb_array_elements(metrics)->>'value')::float as value
    #         FROM unnested
    #     )
    #     SELECT 
    #         EXTRACT(EPOCH FROM h.hour)::bigint * 1000 as timestamp,
    #         ROUND(AVG(p.value)::numeric, 2) as avg_latency,
    #         COUNT(*) as sample_count
    #     FROM hours h
    #     LEFT JOIN parsed p ON h.hour = p.hour
    #     GROUP BY h.hour
    #     ORDER BY h.hour DESC
    #     """,
    #     seven_days_ago, 
    #     current_ts
    #     )

    #     cpu_percent = psutil.cpu_percent()
    #     memory = psutil.virtual_memory()
    #     memory_percent = memory.percent
        
    #     await self.bot.db.execute(
    #         """
    #         INSERT INTO status_metrics (timestamp, cpu_usage, memory_usage)
    #         VALUES ($1, $2, $3)
    #         """,
    #         current_ts, cpu_percent, memory_percent
    #     )
        
    #     ninety_days_ago = current_ts - (90 * 24 * 60 * 60 * 1000)
    #     incidents = await self.bot.db.fetch(
    #         """
    #         SELECT start_time, end_time, severity
    #         FROM incidents
    #         WHERE start_time >= $1 AND severity IN ('major', 'critical')
    #         """,
    #         ninety_days_ago
    #     )
        
    #     total_downtime = sum(
    #         (incident['end_time'] or current_ts) - incident['start_time']
    #         for incident in incidents
    #     )
    #     uptime_period = 90 * 24 * 60 * 60 * 1000 
    #     uptime_percentage = 100 - (total_downtime / uptime_period * 100)

    #     active_incidents = await self.bot.db.fetch(
    #         """
    #         SELECT 
    #             id,
    #             title,
    #             start_time as start,
    #             end_time as end,
    #             status,
    #             severity,
    #             affected_services,
    #             affected_shards,
    #             updates,
    #             created_at
    #         FROM incidents
    #         WHERE 
    #             (start_time >= $1 OR end_time IS NULL)
    #             ORDER BY 
    #                 CASE WHEN end_time IS NULL THEN 0 ELSE 1 END,
    #                 start_time DESC
    #         """,
    #         current_ts - (7 * 24 * 60 * 60 * 1000)  
    #     )
        
    #     current_date = datetime.now(timezone.utc).date()
    #     seconds_in_day = 24 * 60 * 60 
    #     expected_updates_per_day = seconds_in_day // self.metrics_store_interval  
        
    #     return web.json_response({
    #         "shards": [
    #             {
    #                 "guilds": f"{len([guild for guild in self.bot.guilds if guild.shard_id == shard.id]):,}",
    #                 "id": f"{shard.id}",
    #                 "ping": f"{(shard.latency * 1000):.2f}",
    #                 "uptime": f"{int(self.bot.uptime2)}",
    #                 "users": f"{sum(g.member_count for g in self.bot.guilds if g.shard_id == shard.id):,}",
    #             }
    #             for shard in self.bot.shards.values()
    #         ],
    #         "bot": {
    #             "uptime": float(round(uptime_percentage, 2)),
    #             "status_history": [
    #                 {
    #                     "date": row["date"].strftime("%Y-%m-%d"),
    #                     "incidents": (
    #                         row["incidents"] if row["status"] == "has_incidents"
    #                         else []
    #                     ) if row["status"] != "no_data" else None,
    #                     "completion": (
    #                         lambda date=row["date"], updates=row["update_count"]: (
    #                             round(
    #                                 min(100, (updates / (
    #                                     max(1, int(
    #                                         (datetime.now(timezone.utc) - datetime.now(timezone.utc).replace(
    #                                             hour=0, minute=0, second=0, microsecond=0
    #                                         )).total_seconds() / self.metrics_store_interval
    #                                     ))
    #                                     if date == current_date
    #                                     else expected_updates_per_day
    #                                 ) * 100))
    #                                 if updates > 0 else 0,
    #                                 2
    #                             )
    #                         )
    #                     )(),
    #                 }
    #                 for row in historical_data
    #             ]
    #         },
    #         "system": {
    #             "cpu": {
    #                 "current": float(cpu_percent),
    #                 "history": [
    #                     {
    #                         "timestamp": row["timestamp"],
    #                         "value": (
    #                             float(row["avg_cpu"]) if row["sample_count"]
    #                             else (0 if row["timestamp"] > current_ts - 3600000 else None)
    #                         )
    #                     }
    #                     for row in metrics_history
    #                 ]
    #             },
    #             "memory": {
    #                 "current": float(memory_percent),
    #                 "history": [
    #                     {
    #                         "timestamp": row["timestamp"],
    #                         "value": (
    #                             float(row["avg_memory"]) if row["sample_count"]
    #                             else (0 if row["timestamp"] > current_ts - 3600000 else None)
    #                         )
    #                     }
    #                     for row in metrics_history
    #                 ]
    #             },
    #             "latency": {
    #                 "current": float(sum(shard.latency * 1000 for shard in self.bot.shards.values()) / len(self.bot.shards)),
    #                 "history": [
    #                     {
    #                         "timestamp": row["timestamp"],
    #                         "value": (
    #                             float(row["avg_latency"]) if row["avg_latency"] is not None and row["sample_count"]
    #                             else (0 if row["timestamp"] > current_ts - 3600000 else None)
    #                         )
    #                     }
    #                     for row in latency_history
    #                 ]
    #             }
    #         },
    #         "incidents": [
    #             {
    #                 "id": str(incident["id"]),
    #                 "title": incident["title"],
    #                 "start": incident["start"],
    #                 "end": incident["end"],
    #                 "status": incident["status"],
    #                 "severity": incident["severity"],
    #                 "affected_services": incident["affected_services"],
    #                 **({"affected_shards": incident["affected_shards"]} if incident["affected_shards"] else {}),
    #                 "updates": incident["updates"]
    #             }
    #             for incident in active_incidents
    #         ] if active_incidents else []
    #     })

    @route("/status")
    @ratelimit(5, 60)
    async def status(self, request: Request) -> Response:
        return web.json_response(
            {
                "shards": [
                    {
                        "guilds": f"{len([guild for guild in self.bot.guilds if guild.shard_id == shard.id])}",
                        "id": f"{shard.id}",
                        "ping": f"{(shard.latency * 1000):.2f}ms",
                        "uptime": f"{int(self.bot.uptime2)}",
                        "users": f"{sum(guild.member_count for guild in self.bot.guilds if guild.shard_id == shard.id)}",
                    }
                    for shard in self.bot.shards.values()
                ]
            }
        )

    @route("/tickets")
    @ratelimit(5, 60)
    @requires_auth
    async def tickets(self: "Network", request: Request) -> Response:
        ticket_id = request.query.get("id")
        user_id = request.headers.get("User-ID")
        authorization = request.headers.get("Authorization")

        log.info(
            f"Request received for ticket {ticket_id} with Authorization: {authorization} and User-ID: {user_id}"
        )

        if not ticket_id:
            return web.json_response({"error": "Missing ticket ID"}, status=400)
        if not user_id:
            return web.json_response({"error": "Missing User-ID header"}, status=400)

        ticket_path = f"/root/tickets/{ticket_id}.json"
        user_ids_path = f"/root/tickets/{ticket_id}_ids.json"

        if not os.path.isfile(ticket_path):
            log.warning(f"Ticket {ticket_id} not found.")
            return web.json_response({"error": "Ticket not found"}, status=404)

        if not os.path.isfile(user_ids_path):
            log.warning(f"Access list for ticket {ticket_id} not found.")
            return web.json_response(
                {"error": "Access list not found for ticket"}, status=404
            )

        try:
            with open(user_ids_path, "r") as ids_file:
                user_data = json.load(ids_file)
                if user_id not in map(str, user_data.get("ids", [])):
                    log.warning(
                        f"User {user_id} is not authorized to access ticket {ticket_id}."
                    )
                    return web.json_response(
                        {"error": "User not authorized to access this ticket"},
                        status=403,
                    )

            with open(ticket_path, "r") as file:
                ticket_data = json.load(file)

            log.info(f"Ticket {ticket_id} successfully fetched for User-ID: {user_id}.")
            return web.json_response(ticket_data)

        except Exception as e:
            log.error(f"Error reading ticket {ticket_id}: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/tickets", ["POST"])
    @ratelimit(5, 60)
    @requires_auth
    async def create_ticket(self: "Network", request: Request) -> Response:
        try:
            data = await request.json()
            
            if "ticket_id" not in data:
                return web.json_response({"error": "Missing ticket_id"}, status=400)
            if "ticket_data" not in data:
                return web.json_response({"error": "Missing ticket_data"}, status=400)
            if "user_ids" not in data:
                return web.json_response({"error": "Missing user_ids"}, status=400)
                
            ticket_id = data["ticket_id"]
            ticket_data = data["ticket_data"]
            user_ids = data["user_ids"]
            
            ticket_path = f"/root/tickets/{ticket_id}.json"
            user_ids_path = f"/root/tickets/{ticket_id}_ids.json"
            
            if os.path.isfile(ticket_path):
                return web.json_response(
                    {"error": f"Ticket {ticket_id} already exists"}, 
                    status=409
                )
                
            os.makedirs("/root/tickets", exist_ok=True)
            
            try:
                with open(ticket_path, "w") as f:
                    json.dump(ticket_data, f, indent=4)
                    
                with open(user_ids_path, "w") as f:
                    json.dump({"ids": user_ids}, f, indent=4)
                    
                log.info(f"Created ticket {ticket_id} with access for users: {user_ids}")
                
                return web.json_response({
                    "success": True,
                    "message": f"Ticket {ticket_id} created successfully",
                    "ticket_id": ticket_id
                })
                
            except IOError as e:
                log.error(f"Failed to write ticket files: {e}")
                for path in [ticket_path, user_ids_path]:
                    if os.path.exists(path):
                        os.remove(path)
                return web.json_response(
                    {"error": "Failed to save ticket files"}, 
                    status=500
                )
                
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON in request body"}, 
                status=400
            )
        except Exception as e:
            log.error(f"Error creating ticket: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/levels")
    @ratelimit(5, 60)
    @requires_auth
    async def levels(self: "Network", request: Request) -> Response:
        """
        Fetch level information for all members in a guild including user avatar, username, display name, roles, and level roles.
        """
        guild_id = request.headers.get("X-GUILD-ID")
        if not guild_id:
            return web.json_response({"error": "Missing X-GUILD-ID header"}, status=400)

        try:
            guild_id = int(guild_id)
        except ValueError:
            return web.json_response({"error": "Invalid X-GUILD-ID header"}, status=400)

        guild_exists_query = """
        SELECT EXISTS(SELECT 1 FROM level.member WHERE guild_id = $1);
        """
        guild_exists = await self.bot.db.fetchval(guild_exists_query, guild_id)
        if not guild_exists:
            return web.json_response({"error": "Guild ID does not exist"}, status=404)

        level_query = """
        SELECT user_id, xp, level, total_xp, formula_multiplier
        FROM level.member
        INNER JOIN level.config
        ON level.member.guild_id = level.config.guild_id
        WHERE level.member.guild_id = $1;
        """
        level_records = await self.bot.db.fetch(level_query, guild_id)

        users_data = []
        for record in level_records:
            user = self.bot.get_user(record["user_id"])
            if user:
                avatar_url = (
                    str(user.avatar)
                    if user.avatar
                    else (
                        str(user.default_avatar_url)
                        if user.default_avatar_url
                        else None
                    )
                )

                users_data.append(
                    {
                        "user_id": record["user_id"],
                        "xp": record["xp"],
                        "level": record["level"],
                        "total_xp": record["total_xp"],
                        "max_xp": self.required_xp(
                            record["level"] + 1, record["formula_multiplier"]
                        ),
                        "avatar_url": avatar_url,
                        "username": user.name if user.name else "Unknown",
                        "display_name": (
                            user.display_name if user.display_name else "Unknown"
                        ),
                    }
                )

        level_roles_query = """
        SELECT role_id, level
        FROM level.role
        WHERE guild_id = $1
        """
        level_roles_records = await self.bot.db.fetch(level_roles_query, guild_id)

        level_roles = []
        guild = self.bot.get_guild(guild_id)
        for role in level_roles_records:
            discord_role = discord.utils.get(guild.roles, id=role["role_id"])
            if discord_role:
                level_roles.append(
                    {
                        "role_id": role["role_id"],
                        "level": role["level"],
                        "role_name": discord_role.name,
                        "hex_color": (
                            str(discord_role.color) if discord_role.color else None
                        ),
                    }
                )

        response_data = {
            "guild_id": guild_id,
            "guild_name": guild.name if guild else "Unknown Guild",
            "level_roles": level_roles,
            "users": users_data,
        }

        return web.json_response(response_data)

    @route("/ws/music/{guild_id}", ["GET"])
    async def music_websocket(self, request: web.Request) -> web.WebSocketResponse:
        guild_id = str(request.match_info["guild_id"])
        auth_token = request.query.get("auth")
        
        if not auth_token:
            log.warning(f"[WS] Rejected connection for guild {guild_id}: missing auth token")
            return web.Response(status=401, text="Missing authorization")
        
        log.info(f"[WS] Incoming request for guild {guild_id} from {request.remote}")
        
        if guild_id in self.ws_connections and auth_token in self.ws_connections[guild_id]:
            try:
                existing_ws = self.ws_connections[guild_id][auth_token]
                if not existing_ws.closed:
                    await existing_ws.close(code=1000, message=b'New connection requested')
                    log.info(f"[WS] Closed existing connection for guild {guild_id} with same auth token")
            except Exception as e:
                log.error(f"[WS] Error closing existing connection for guild {guild_id}: {e}")
            finally:
                if guild_id in self.ws_connections:
                    self.ws_connections[guild_id].pop(auth_token, None)
                    if not self.ws_connections[guild_id]:
                        del self.ws_connections[guild_id]
        
        ws = web.WebSocketResponse(
            heartbeat=30,
            protocols=['json'],
            autoping=True,
            timeout=30
        )
        
        if not ws.can_prepare(request):
            log.error(f"[WS] Cannot prepare WebSocket for guild {guild_id}: incompatible protocol")
            return web.Response(status=400, text="Invalid WebSocket request")
        
        try:
            log.info(f"[WS] Attempting to prepare connection for guild {guild_id}")
            await ws.prepare(request)
            log.info(f"[WS] Connection established for guild {guild_id}")
            
            if guild_id not in self.ws_connections:
                self.ws_connections[guild_id] = {}
            
            self.ws_connections[guild_id][auth_token] = ws
            
            try:
                auth_token = request.query.get("auth")
                if not auth_token:
                    log.warning(f"[WS] Auth failed for guild {guild_id}: missing token")
                    await ws.close(code=4001, message=b"Missing authorization")
                    return ws

                await ws.send_json({
                    "type": "HELLO",
                    "data": {
                        "heartbeat_interval": 30000
                    }
                })

                update_task = asyncio.create_task(self.send_state_updates(ws, guild_id))
                log.info(f"Started state update task for guild {guild_id}")

                try:
                    async for msg in ws:
                        if msg.type == WSMsgType.CLOSE:
                            log.info(f"WebSocket connection closing for guild {guild_id}")
                            break
                        elif msg.type == WSMsgType.ERROR:
                            log.error(f"WebSocket error for guild {guild_id}: {ws.exception()}")
                            break
                        elif msg.type == WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                
                                if data.get("type") == "HELLO":
                                    await ws.send_json({"type": "READY"})
                                    continue

                                if data.get("type") == "PING":
                                    await ws.send_json({"type": "PONG"})
                                    continue

                                command = data.get("command")
                                if not command:
                                    continue

                                log.info(
                                    f"Received WebSocket command '{command}' for guild {guild_id}"
                                )

                                audio_cog = self.bot.get_cog("Audio")
                                if not audio_cog:
                                    log.error(
                                        f"Audio cog not available for guild {guild_id}"
                                    )
                                    await ws.send_json(
                                        {"error": "Audio system not available"}
                                    )
                                    continue

                                guild = self.bot.get_guild(int(guild_id))
                                if not guild:
                                    log.error(f"Guild {guild_id} not found")
                                    await ws.send_json({"error": "Guild not found"})
                                    continue

                                voice_client = guild.voice_client
                                if not voice_client:
                                    log.error(f"No voice client for guild {guild_id}")
                                    await ws.send_json({"error": "Not connected to voice"})
                                    continue

                                if command == "play":
                                    if not voice_client.is_playing():
                                        await voice_client.resume()
                                        await ws.send_json({"network": "resumed"})
                                        log.info(f"Resumed playback for guild {guild_id}")

                                elif command == "pause":
                                    if voice_client.is_playing():
                                        await voice_client.pause()
                                        await ws.send_json({"network": "paused"})
                                        log.info(f"Paused playback for guild {guild_id}")

                                elif command == "skip":
                                    await voice_client.stop()
                                    await ws.send_json({"network": "skipped"})
                                    log.info(f"Skipped track for guild {guild_id}")

                                elif command == "seek":
                                    position = data.get("position")
                                    if position is not None:
                                        await voice_client.seek(position)
                                        await ws.send_json(
                                            {"network": "seeked", "position": position}
                                        )
                                        log.info(
                                            f"Seeked to position {position} for guild {guild_id}"
                                        )

                                elif command == "volume":
                                    volume = data.get("volume")
                                    if volume is not None:
                                        await voice_client.set_volume(volume)
                                        await ws.send_json(
                                            {"network": "volume_changed", "volume": volume}
                                        )
                                        log.info(
                                            f"Changed volume to {volume} for guild {guild_id}"
                                        )

                                elif command == "shuffle":
                                    if voice_client.queue:
                                        voice_client.queue.shuffle()
                                        await ws.send_json({"network": "shuffled"})
                                        log.info(f"Shuffled queue for guild {guild_id}")

                                elif command == "repeat":
                                    mode = data.get("mode")
                                    if mode in ["track", "queue", "off"]:
                                        if mode == "track":
                                            voice_client.queue.set_loop_mode(LoopMode.TRACK)
                                        elif mode == "queue":
                                            voice_client.queue.set_loop_mode(LoopMode.QUEUE)
                                        else:
                                            voice_client.queue.disable_loop()
                                        await ws.send_json(
                                            {"network": "repeat_changed", "mode": mode}
                                        )
                                        log.info(
                                            f"Changed repeat mode to {mode} for guild {guild_id}"
                                        )

                                elif command == "play_index":
                                    index = data.get("index")
                                    if index is not None and voice_client.queue:
                                        if 0 <= index < len(voice_client.queue):
                                            track = voice_client.queue[index]
                                            voice_client.queue.remove(track)
                                            voice_client.queue._queue.insert(0, track)
                                            await voice_client.stop()
                                            await ws.send_json(
                                                {"network": "playing_index", "index": index}
                                            )
                                            log.info(
                                                f"Playing track at index {index} for guild {guild_id}"
                                            )

                                elif command == "next":
                                    if voice_client.queue:
                                        next_track = voice_client.queue[0]
                                        await voice_client.stop()
                                        await ws.send_json(
                                            {
                                                "network": "next",
                                                "track": {
                                                    "title": next_track.title,
                                                    "artist": next_track.author,
                                                    "duration": next_track.length,
                                                    "thumbnail": next_track.thumbnail
                                                    or self.default_thumbnail,
                                                    "uri": next_track.uri,
                                                },
                                            }
                                        )
                                        log.info(f"Playing next track for guild {guild_id}")

                                elif command == "previous":
                                    await ws.send_json({"network": "previous_not_supported"})
                                    log.info(
                                        f"Previous track requested but not supported for guild {guild_id}"
                                    )

                            except Exception as e:
                                log.error(f"Error handling command for guild {guild_id}: {e}")
                                await ws.send_json({"error": str(e)})

                except Exception as e:
                    log.error(f"WebSocket error for guild {guild_id}: {e}")
                finally:
                    if 'update_task' in locals():
                        update_task.cancel()
                    return ws
            except Exception as e:
                log.error(f"WebSocket error for guild {guild_id}: {e}")
                return ws
        except Exception as e:
            log.error(f"Failed to prepare WebSocket for guild {guild_id}: {e}")
            return ws

    async def send_state_updates(self, ws, guild_id):
        """Separate task for sending state updates"""
        try:
            while not ws.closed:
                guild = self.bot.get_guild(int(guild_id))
                state_data = {
                    "type": "STATE_UPDATE",
                    "data": {
                        "current": None,
                        "position": 0,
                        "queue": [],
                        "controls": {
                            "volume": 100,
                            "isPlaying": False,
                            "repeat": "off",
                            "shuffle": False
                        }
                    }
                }

                if guild and guild.voice_client:
                    voice_client = guild.voice_client
                    state_data["data"]["controls"].update({
                        "volume": voice_client.volume,
                        "isPlaying": not voice_client.is_paused,
                        "repeat": voice_client.queue.loop_mode.value if voice_client.queue and voice_client.queue.loop_mode else "off",
                        "shuffle": bool(voice_client.queue and voice_client.queue.shuffle)
                    })

                    if voice_client.current:
                        state_data["data"]["current"] = {
                            "title": voice_client.current.title,
                            "artist": voice_client.current.author,
                            "duration": voice_client.current.length,
                            "thumbnail": "https://upload.wikimedia.org/wikipedia/en/5/51/Kendrick_Lamar_-_Damn.png",
                            "uri": voice_client.current.uri,
                            "position": voice_client.position,
                            "is_playing": not voice_client.is_paused
                        }
                        state_data["data"]["queue"] = [
                            {
                                "title": t.title,
                                "artist": t.author,
                                "duration": t.length,
                                "thumbnail": "https://upload.wikimedia.org/wikipedia/en/5/51/Kendrick_Lamar_-_Damn.png",
                                "uri": t.uri
                            }
                            for t in (voice_client.queue or [])
                        ]

                log.debug(f"Sending state update for guild {guild_id}")
                await ws.send_json(state_data)
                await asyncio.sleep(1)
                
        except Exception as e:
            log.error(f"Error in state updates for guild {guild_id}: {e}")

    @route("/spotify/auth", ["POST"])
    @ratelimit(5, 60)
    @requires_not_auth
    async def spotify_auth(self: "Network", request: Request) -> Response:
        try:
            data = await request.json()
            
            required_fields = {
                "user_id": data.get("user_id"),
                "spotify_access_token": data.get("spotify_access_token"), 
                "spotify_refresh_token": data.get("spotify_refresh_token"),
                "expires_in": data.get("expires_in"),
                "spotify_id": data.get("spotify_id")
            }

            if missing := [k for k, v in required_fields.items() if not v]:
                return web.json_response(
                    {"error": f"Missing required fields: {', '.join(missing)}"},
                    status=400
                )

            current_time = datetime.now(timezone.utc)
            expires_at = (current_time + timedelta(seconds=int(required_fields["expires_in"])))
            
            expires_at_ts = expires_at.replace(tzinfo=None)

            await self.bot.db.execute(
                """
                INSERT INTO user_spotify (
                    user_id, access_token, refresh_token, token_expires_at, spotify_id
                ) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id) DO UPDATE SET
                    access_token = EXCLUDED.access_token,
                    refresh_token = EXCLUDED.refresh_token, 
                    token_expires_at = EXCLUDED.token_expires_at,
                    spotify_id = EXCLUDED.spotify_id
                """,
                int(required_fields["user_id"]),
                required_fields["spotify_access_token"],
                required_fields["spotify_refresh_token"], 
                expires_at_ts, 
                required_fields["spotify_id"]
            )

            return web.json_response({"success": True})
            
        except Exception as e:
            log.error(f"Error processing Spotify auth: {str(e)}", exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    async def handle_options(self, request):
        return web.Response(status=200)

    @route("/lastfm/auth", ["POST"])
    @ratelimit(5, 60)
    @requires_not_auth
    async def lastfm_auth(self: "Network", request: Request) -> Response:
        try:
            data = await request.json()
            
            required_fields = {
                "user_id": data.get("user_id"),
                "access_token": data.get("access_token"),
                "username": data.get("username")
            }

            if missing := [k for k, v in required_fields.items() if not v]:
                return web.json_response(
                    {"error": f"Missing required fields: {', '.join(missing)}"},
                    status=400
                )

            await self.bot.db.execute(
                """
                INSERT INTO lastfm.config (
                    user_id, access_token, username, web_authentication
                ) VALUES ($1, $2, $3, true)
                ON CONFLICT (user_id) DO UPDATE SET
                    access_token = EXCLUDED.access_token,
                    username = EXCLUDED.username,
                    web_authentication = true
                """,
                int(required_fields["user_id"]),
                required_fields["access_token"],
                required_fields["username"]
            )

            return web.json_response({"success": True})
            
        except Exception as e:
            log.error(f"Error processing Last.fm auth: {str(e)}", exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    async def can_send_failure_notification(self, user_id: int) -> bool:
        """Check if we can send a failure notification to the user (max 2 per hour)"""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        
        self.failed_payment_notifications[user_id] = [
            timestamp for timestamp in self.failed_payment_notifications[user_id]
            if timestamp > hour_ago
        ]
        
        return len(self.failed_payment_notifications[user_id]) < 2

    @route("/stripe-webhook", ["POST"])
    @ratelimit(10, 60)
    async def stripe_webhook(self: "Network", request: Request) -> Response:
        try:
            payload = await request.text()
            sig_header = request.headers.get('Stripe-Signature')

            try:
                event = stripe.Webhook.construct_event(
                    payload, sig_header, ""
                )
            except ValueError:
                log.error("Invalid Stripe payload received")
                return web.json_response({"error": "Invalid payload"}, status=400)
            except stripe.error.SignatureVerificationError:
                log.error("Invalid Stripe signature")
                return web.json_response({"error": "Invalid signature"}, status=400)

            log_channel = self.bot.get_channel(1319933684576550922)
            
            if event['type'] == 'checkout.session.completed':
                session = event['data']['object']
                discord_id = None
                if 'custom_fields' in session:
                    for field in session['custom_fields']:
                        if field['key'] == 'useriddiscord':
                            discord_id = field['text']['value']
                            break

                if discord_id:
                    try:
                        user = await self.bot.fetch_user(int(discord_id))
                        guild = self.bot.get_guild(892675627373699072)
                        member = await guild.fetch_member(int(discord_id))

                        if session['payment_link'] == '':
                            await self.bot.db.execute(
                                """
                                INSERT INTO instances 
                                (user_id, payment_id, amount, purchased_at, expires_at, status, email)
                                VALUES ($1, $2, $3, NOW(), NOW() + INTERVAL '30 days', 'pending', $4)
                                """,
                                int(discord_id),
                                session['payment_intent'],
                                session['amount_total'] / 100,
                                session['customer_details']['email']
                            )

                            guild = self.bot.get_guild(892675627373699072)
                            if guild:
                                member = await guild.fetch_member(int(discord_id))
                                if member:
                                    role = guild.get_role(1320428924215496704)
                                    if role and role not in member.roles:
                                        await member.add_roles(role, reason="Instance purchased")

                            embed = Embed(
                                title="Thank You for Purchasing an Evict Instance!",
                                description=(
                                    "Your instance purchase has been processed successfully! 🎉\n\n"
                                    "To complete the setup process, there's a small monthly hosting fee of $3 to "
                                    "keep your instance running smoothly. This helps us maintain the infrastructure "
                                    "and ensure high availability for your bot.\n\n"
                                    "**Next Steps:**\n"
                                    "- Complete the hosting subscription: [Click Here](https://buy.stripe.com/)\n"
                                    "- Once subscribed, your instance will be ready for setup\n"
                                    "- You'll receive a message on how to edit your instance\n\n"
                                    "*Note: The hosting subscription ensures your "
                                    "instance stays online and receives regular updates and maintenance.*"
                                ),
                                color=0x2ecc71
                            )
                            embed.set_thumbnail(url=self.bot.user.display_avatar.url)
                            await user.send(embed=embed)
                        
                            if log_channel:
                                log_embed = Embed(
                                    title="New Instance Purchase",
                                    description=(
                                        f"User: {user.mention} (`{user.id}`)\n"
                                        f"Amount: ${session['amount_total'] / 100:.2f} {session['currency'].upper()}\n"
                                        f"Payment ID: `{session['payment_intent']}`"
                                    ),
                                    color=0x2ecc71,
                                    timestamp=datetime.now(timezone.utc)
                                )
                                await log_channel.send(embed=log_embed)

                        elif session['payment_link'] == '':
                            instance = await self.bot.db.fetchrow(
                                """
                                SELECT * FROM instances 
                                WHERE user_id = $1 AND status = 'pending'
                                """,
                                int(discord_id)
                            )
                            
                            if not instance:
                                embed = Embed(
                                    title="Hosting Subscription Error",
                                    description=(
                                        "Oops! It looks like you haven't purchased an instance yet.\n\n"
                                        "Please purchase an instance first before activating the hosting subscription:\n"
                                        "[Purchase Instance](https://buy.stripe.com/)\n\n"
                                        "If you believe this is an error, please contact our support team."
                                    ),
                                    color=0xff0000
                                )
                                embed.set_thumbnail(url=self.bot.user.display_avatar.url)
                                await user.send(embed=embed)
                                return web.json_response({"error": "No pending instance found"}, status=400)

                            await self.bot.db.execute(
                                """
                                UPDATE instances 
                                SET status = 'active'
                                WHERE user_id = $1 AND status = 'pending'
                                """,
                                int(discord_id)
                            )

                            embed = Embed(
                                title="Instance Hosting Subscription Activated!",
                                description=(
                                    "Your instance hosting subscription has been activated! 🎉\n\n"
                                    "To set up your instance, use the following command:\n"
                                    "`;instance setup <name> <prefix>`\n\n"
                                    "After setup, you can customize your instance using:\n"
                                    "- `[prefix]customize` - Change bot appearance\n"
                                    "- `[prefix]activity` - Set bot status/activity\n\n"
                                    "Want custom commands? Create a ticket in our "
                                    "[support server](https://discord.gg/evict)\n\n"
                                    "If you need any assistance, our support team is ready to help!"
                                ),
                                color=0x2ecc71
                            )
                            embed.set_thumbnail(url=self.bot.user.display_avatar.url)
                            await user.send(embed=embed)

                            if log_channel:
                                log_embed = Embed(
                                    title="Instance Hosting Subscription Activated",
                                    description=(
                                        f"User: {user.mention} (`{user.id}`)\n"
                                        f"Amount: ${session['amount_total'] / 100:.2f} {session['currency'].upper()}\n"
                                        f"Payment ID: `{session['payment_intent']}`"
                                    ),
                                    color=0x2ecc71,
                                    timestamp=datetime.now(timezone.utc)
                                )
                                await log_channel.send(embed=log_embed)

                        else:
                            check = await self.bot.db.fetchrow(
                                """
                                SELECT user_id 
                                FROM donators 
                                WHERE user_id = $1
                                """,
                                int(discord_id)
                            )
                            if check is None:
                                await self.bot.db.execute(
                                    """
                                    INSERT INTO donators 
                                    VALUES ($1)
                                    """, 
                                    int(discord_id)
                                )

                            embed = Embed(
                                title="Thank You for Supporting evict!",
                                description=(
                                    "Your donation has been received and processed successfully! 🎉\n\n"
                                    "You now have access to premium features including:\n"
                                    "- Custom bot reskins\n"
                                    "- Extended limits for OpenAI features\n"
                                    "- Enhanced transcription capabilities\n"
                                    "- Priority support\n\n"
                                    "Thank you for helping keep evict running! ❤️"
                                ),
                                color=0x2ecc71
                            )
                            embed.set_thumbnail(url=self.bot.user.display_avatar.url)
                            await user.send(embed=embed)

                            role = guild.get_role(1318054098666389534)
                            if role and role not in member.roles:
                                await member.add_roles(role, reason="Donation received")

                            if log_channel:
                                log_embed = Embed(
                                    title="New Donation Received",
                                    description=(
                                        f"User: {user.mention} (`{user.id}`)\n"
                                        f"Amount: ${session['amount_total'] / 100:.2f} {session['currency'].upper()}\n"
                                        f"Payment ID: `{session['payment_intent']}`"
                                    ),
                                    color=0x2ecc71,
                                    timestamp=datetime.now(timezone.utc)
                                )
                                await log_channel.send(embed=log_embed)

                        return web.json_response({"success": True})

                    except Exception as e:
                        log.error(f"Failed to process donation for user {discord_id}: {e}")
                        if log_channel:
                            error_embed = Embed(
                                title="Donation Processing Failed",
                                description=(
                                    f"Failed to process donation for Discord ID: `{discord_id}`\n"
                                    f"Amount: ${session['amount_total'] / 100:.2f} {session['currency'].upper()}\n"
                                    f"Payment ID: `{session['payment_intent']}`\n"
                                    f"Error: ```{str(e)}```"
                                ),
                                color=0xff0000,
                                timestamp=datetime.now(timezone.utc)
                            )
                            await log_channel.send(embed=error_embed)
                        return web.json_response(
                            {"error": f"Failed to process donation: {str(e)}"}, 
                            status=500
                        )
                else:
                    if log_channel:
                        error_embed = Embed(
                            title="Donation Processing Failed",
                            description=(
                                "Payment received but no Discord ID provided\n"
                                f"Amount: ${session['amount_total'] / 100:.2f} {session['currency'].upper()}\n"
                                f"Payment ID: `{session['payment_intent']}`\n"
                                f"Raw session data: ```{json.dumps(session, indent=2)}```"
                            ),
                            color=0xff0000,
                            timestamp=datetime.now(timezone.utc)
                        )
                        await log_channel.send(embed=error_embed)
                    return web.json_response(
                        {"error": "Missing discord_user_id in metadata"}, 
                        status=400
                    )

            elif event['type'] == 'checkout.session.expired':
                session = event['data']['object']
                discord_id = session['metadata'].get('discord_user_id')
                if log_channel:
                    embed = Embed(
                        title="Checkout Session Expired",
                        description=(
                            f"Discord ID: `{discord_id if discord_id else 'Not provided'}`\n"
                            f"Session ID: `{session['id']}`"
                        ),
                        color=0xffa500,
                        timestamp=datetime.now(timezone.utc)
                    )
                    await log_channel.send(embed=embed)

            elif event['type'] == 'payment_intent.payment_failed':
                intent = event['data']['object']
                discord_id = None
                
                if 'custom_fields' in intent:
                    for field in intent['custom_fields']:
                        if field['key'] == 'useriddiscord':
                            discord_id = field['text']['value']
                            break
                
                if log_channel:
                    embed = Embed(
                        title="Payment Failed",
                        description=(
                            f"Discord ID: `{discord_id if discord_id else 'Not provided'}`\n"
                            f"Payment ID: `{intent['id']}`\n"
                            f"Error: `{intent['last_payment_error']['message'] if intent.get('last_payment_error') else 'Unknown error'}`"
                        ),
                        color=0xff0000,
                        timestamp=datetime.now(timezone.utc)
                    )
                    await log_channel.send(embed=embed)

                if discord_id:
                    try:
                        user_id = int(discord_id)
                        if await self.can_send_failure_notification(user_id):
                            user = await self.bot.fetch_user(user_id)
                            error_message = (
                                intent['last_payment_error']['message'] 
                                if intent.get('last_payment_error') 
                                else 'Unknown error'
                            )
                            
                            embed = Embed(
                                title="Payment Failed",
                                description=(
                                    "Your payment to evict could not be processed.\n\n"
                                    f"Reason: {error_message}\n\n"
                                    "You can try again with a different payment method or contact your bank "
                                    "if you believe this is an error."
                                ),
                                color=0xff0000,
                                timestamp=datetime.now(timezone.utc)
                            )
                            embed.set_thumbnail(url=self.bot.user.display_avatar.url)
                            await user.send(embed=embed)
                            
                            self.failed_payment_notifications[user_id].append(datetime.now(timezone.utc))
                    except Exception as e:
                        log.error(f"Failed to send payment failure notification to user {discord_id}: {e}")

            return web.json_response({"success": True})

        except Exception as e:
            log.error(f"Error processing webhook: {e}")
            if log_channel:
                error_embed = Embed(
                    title="Webhook Processing Error",
                    description=f"Error: ```{str(e)}```",
                    color=0xff0000,
                    timestamp=datetime.now(timezone.utc)
                )
                await log_channel.send(embed=error_embed)
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/lovense/callback", ["GET"])
    @ratelimit(10, 60)
    async def lovense_callback(self, request: Request) -> Response:
        """Handle Lovense OAuth callback"""
        try:
            code = request.query.get("code")
            state = request.query.get("state")
            
            if not code or not state:
                return web.json_response(
                    {"error": "Missing code or state parameter"}, 
                    status=400
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.lovense.com/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "client_id": "-X1p4MV3pUVZoygskhfrkisx69F7y2LJJzglk_d51s-rackNZPcogzu48d5Z4EHD",
                        "client_secret": "b5c0e61d3ff07bf8"
                    }
                ) as resp:
                    if resp.status != 200:
                        return web.json_response(
                            {"error": "Failed to get access token"}, 
                            status=400
                        )
                    
                    data = await resp.json()
                    access_token = data["access_token"]
                    
                    async with session.get(
                        "https://api.lovense.com/api/v2/user/devices",
                        headers={"Authorization": f"Bearer {access_token}"}
                    ) as resp:
                        if resp.status != 200:
                            return web.json_response(
                                {"error": "Failed to get devices"}, 
                                status=400
                            )
                        
                        devices = await resp.json()
                        
                        if not devices:
                            return web.json_response(
                                {"error": "No devices found"}, 
                                status=400
                            )
                        
                        device = devices[0]  
                        
                        connection = await self.bot.db.fetchrow(
                            """
                            SELECT guild_id, user_id 
                            FROM lovense_connections 
                            WHERE token = $1 
                            AND expires_at > CURRENT_TIMESTAMP
                            """,
                            state
                        )
                        
                        if not connection:
                            return web.json_response(
                                {"error": "Invalid or expired connection token"}, 
                                status=400
                            )
                        
                        guild_id = connection['guild_id']
                        user_id = connection['user_id']
                        
                        await self.bot.db.execute(
                            """
                            INSERT INTO lovense_devices 
                            (guild_id, user_id, device_id, device_type, access_token)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (guild_id, user_id) 
                            DO UPDATE SET 
                                device_id = $3,
                                device_type = $4,
                                access_token = $5,
                                last_active = CURRENT_TIMESTAMP
                            """,
                            int(guild_id), int(user_id), 
                            device["id"], device["type"],
                            access_token
                        )

                        await self.bot.db.execute(
                            """
                            UPDATE lovense_connections
                            SET completed_at = CURRENT_TIMESTAMP,
                                device_id = $1
                            WHERE token = $2
                            """,
                            device["id"], state
                        )

                        try:
                            user = self.bot.get_user(int(user_id))
                            if user:
                                await user.send(
                                    f"Successfully connected your {device['type']}! "
                                    "You can now use Lovense commands in the server."
                                )
                        except:
                            pass

            return web.Response(
                status=302,
                headers={
                    "Location": "https://evict.bot/connected" 
                }
            )

        except Exception as e:
            log.error(f"Error in Lovense callback: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/users/presence", ["GET"])
    @ratelimit(10, 60)
    async def get_users_presence(self: "Network", request: Request) -> Response:
        try:
            guild = self.bot.get_guild(892675627373699072)
            if not guild:
                return web.json_response({"success": False, "error": "Guild not found"}, status=404)

            tracked_roles = [
                guild.get_role(1265473601755414528), 
                guild.get_role(1264110559989862406),
                guild.get_role(1323255508609663098)
            ]

            response_data = {"data": [], "success": True}

            for role in tracked_roles:
                if not role:
                    continue

                for member in role.members:
                    activities = []
                    spotify_data = None
                    
                    for activity in member.activities:
                        if activity.name == "Spotify":
                            spotify_data = {
                                "timestamps": getattr(activity, "timestamps", {}),
                                "album": getattr(activity, "album", None),
                                "album_art_url": f"https://i.scdn.co/image/{activity.album_cover_url.split(':', 1)[1]}" if getattr(activity, "album_cover_url", None) else None,
                                "artist": activity.artist,
                                "song": activity.title,
                                "track_id": activity.track_id
                            }
                            
                        activity_data = {
                            "flags": getattr(activity, "flags", 0),
                            "id": getattr(activity, "application_id", None), 
                            "name": activity.name,
                            "type": activity.type.value,
                            "state": getattr(activity, "state", None),
                            "details": getattr(activity, "details", None),
                            "created_at": int(activity.created_at.timestamp() * 1000) if hasattr(activity, "created_at") else None,
                            "timestamps": getattr(activity, "timestamps", {}),
                            "assets": {
                                "large_image": getattr(activity.assets, "large_image", None) if hasattr(activity, "assets") else None,
                                "large_text": getattr(activity.assets, "large_text", None) if hasattr(activity, "assets") else None,
                                "small_image": getattr(activity.assets, "small_image", None) if hasattr(activity, "assets") else None,
                                "small_text": getattr(activity.assets, "small_text", None) if hasattr(activity, "assets") else None
                            } if hasattr(activity, "assets") else None
                        }

                        if activity.name == "Spotify":
                            activity_data["sync_id"] = activity.track_id
                            activity_data["party"] = {"id": f"spotify:{member.id}"}
                        
                        activities.append(activity_data)

                    public_flags = member.public_flags.value
                    badges = []
                    
                    if public_flags & (1 << 0): badges.append("Discord_Staff")
                    if public_flags & (1 << 1): badges.append("Discord_Partner")
                    if public_flags & (1 << 2): badges.append("HypeSquad_Events")
                    if public_flags & (1 << 3): badges.append("Bug_Hunter_Level_1")
                    if public_flags & (1 << 6): badges.append("House_Bravery")
                    if public_flags & (1 << 7): badges.append("Early_Supporter")
                    if public_flags & (1 << 8): badges.append("House_Balance")
                    if public_flags & (1 << 9): badges.append("House_Brilliance")
                    if public_flags & (1 << 14): badges.append("Bug_Hunter_Level_2")
                    if public_flags & (1 << 16): badges.append("Verified_Bot_Developer")
                    if public_flags & (1 << 17): badges.append("Early_Verified_Bot_Developer")
                    if public_flags & (1 << 22): badges.append("Active_Developer")
                    
                    if member.premium_since:
                        badges.append("Discord_Nitro")
                        badges.append("Nitro_Boost")
                        
                        # Remove all the monthly checks
                        # boost_duration = (datetime.now(timezone.utc) - member.premium_since).days
                        # boost_months = boost_duration // 30
                        
                        # if boost_months >= 1: badges.append("Booster_1Month")
                        # ... remove all the monthly badge checks ...

                    links = await self.bot.db.fetch(
                        """
                        SELECT type, url 
                        FROM user_links 
                        WHERE user_id = $1
                        """,
                        member.id
                    )
                    
                    user_data = {
                        "kv": {},
                        "discord_user": {
                            "id": str(member.id),
                            "username": member.name,
                            "avatar": member.avatar.key if member.avatar else None,
                            "discriminator": member.discriminator,
                            "clan": {
                                "tag": None,
                                "identity_guild_id": "",
                                "badge": None,
                                "identity_enabled": True
                            },
                            "avatar_decoration_data": {
                                "sku_id": member.avatar_decoration_sku_id,
                                "asset": member.avatar_decoration.key if member.avatar_decoration else None,
                                "expires_at": None 
                            } if member.avatar_decoration else None,
                            "bot": member.bot,
                            "global_name": member.global_name,
                            "primary_guild": {
                                "tag": None,
                                "identity_guild_id": None,
                                "badge": None,
                                "identity_enabled": True
                            },
                            "display_name": member.display_name,
                            "public_flags": public_flags,
                            "badges": badges,
                            "roles": [str(role.id)],
                            "links": {
                                link['type']: link['url']
                                for link in links
                            } if links else {}
                        },
                        "activities": activities,
                        "discord_status": str(member.status),
                        "active_on_discord_web": member.web_status != Status.offline,
                        "active_on_discord_desktop": member.desktop_status != Status.offline,
                        "active_on_discord_mobile": member.mobile_status != Status.offline,
                        "listening_to_spotify": bool(spotify_data),
                        "spotify": spotify_data
                    }

                    response_data["data"].append(user_data)

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error in get_users_presence: {e}", exc_info=True)
            return web.json_response(
                {"success": False, "error": "Internal server error"}, 
                status=500
            )

    @route("/avatars/{user_id}")
    @ratelimit(10, 60)
    @requires_auth
    async def avatars(self, request: Request) -> Response:
        """Get avatar history for a user"""
        try:
            user_id = int(request.match_info["user_id"])
            
            avatars = await self.bot.db.fetch(
                """
                SELECT avatar_url, timestamp::text
                FROM avatar_history
                WHERE user_id = $1 AND deleted_at IS NULL
                ORDER BY timestamp DESC
                """,
                user_id
            )

            if not avatars:
                return web.json_response({"error": "No avatar history found"}, status=404)

            user = self.bot.get_user(user_id)
            if not user:
                try:
                    user = await self.bot.fetch_user(user_id)
                except:
                    return web.json_response({"error": "User not found"}, status=404)

            return web.json_response({
                "user": {
                    "id": str(user.id),
                    "name": user.name,
                    "discriminator": user.discriminator if hasattr(user, "discriminator") else None,
                    "avatar": str(user.avatar.url) if user.avatar else None,
                    "display_name": user.display_name if hasattr(user, "display_name") else user.name
                },
                "avatars": [
                    {
                        "url": avatar["avatar_url"],
                        "timestamp": avatar["timestamp"]
                    }
                    for avatar in avatars
                ],
                "total": len(avatars)
            })

        except ValueError:
            return web.json_response({"error": "Invalid user ID"}, status=400)
        except Exception as e:
            log.error(f"Error fetching avatar history: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/playing/{user_id}")
    @ratelimit(30, 60)
    async def currently_playing(self, request: Request) -> Response:
        """Get currently playing track info for a specific user"""
        try:
            user_id = int(request.match_info["user_id"])
            audio_cog = self.bot.get_cog("Audio")
            
            if not audio_cog:
                return web.json_response({"error": "Audio cog not loaded"}, status=503)

            async def get_track_info(track):
                title = track.title
                artist = track.author

                if "feat." in title.lower():
                    title = title.split("feat.")[0].strip()
                
                title = (title
                    .replace("(Official Music Video)", "")
                    .replace("(Official Video)", "")
                    .replace("(Official Audio)", "")
                    .replace("(Lyric Video)", "")
                    .replace("(Lyrics)", "")
                    .replace("[Official Video]", "")
                    .strip())

                if " - " in title:
                    parts = title.split(" - ", 1)
                    if len(parts) == 2:
                        artist, title = parts
                
                artist = artist.strip()
                
                async with aiohttp.ClientSession() as session:
                    search_url = "https://api.deezer.com/search"
                    params = {
                        "q": f'artist:"{artist}" track:"{title}"'
                    }
                    
                    async with session.get(search_url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get('data'):
                                track_data = data['data'][0]
                                return {
                                    "title": track.title, 
                                    "artist": artist,
                                    "album": track_data['album']['title'],
                                    "album_art": track_data['album']['cover_xl'],
                                    "uri": track.uri,
                                    "length": track.length,
                                    "is_stream": track.is_stream,
                                    "requester_id": str(track.requester.id) if track.requester else None
                                }
                
                return {
                    "title": track.title,
                    "artist": artist,
                    "album": None,
                    "album_art": None,
                    "uri": track.uri,
                    "length": track.length,
                    "is_stream": track.is_stream,
                    "requester_id": str(track.requester.id) if track.requester else None
                }

            playing_data = None
            for guild in self.bot.guilds:
                voice_client = guild.voice_client
                if not voice_client or not voice_client.is_playing: 
                    continue
                    
                voice_members = voice_client.channel.members
                if any(m.id == user_id for m in voice_members):
                    track = voice_client.current
                    if track:
                        current_track_info = await get_track_info(track)
                        queue = [t for t in voice_client.queue]
                        queue_data = []
                        
                        for t in queue[:10]:
                            queue_data.append(await get_track_info(t))

                        playing_data = {
                            "current": {
                                **current_track_info,
                                "position": voice_client.position
                            },
                            "queue": queue_data,
                            "queue_length": len(voice_client.queue),
                            "guild_id": str(guild.id),
                            "channel_id": str(voice_client.channel.id),
                            "voice_state": {
                                "volume": voice_client.volume,
                                "paused": voice_client.is_paused,
                                "loop_mode": str(voice_client.queue.loop_mode) if hasattr(voice_client.queue, 'loop_mode') else "NONE",
                                "auto_play": voice_client.auto_play if hasattr(voice_client, 'auto_play') else False
                            }
                        }
                        break

            if playing_data:
                return web.json_response(playing_data)
            else:
                return web.json_response({"error": "User not in any voice channel with active playback"}, status=404)

        except ValueError:
            return web.json_response({"error": "Invalid user ID"}, status=400)
        except Exception as e:
            log.error(f"Error getting currently playing info: {e}", exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/login", ["POST"])
    @ratelimit(5, 60)
    async def login(self, request: Request) -> Response:
        """Handle login and access token creation"""
        if "X-Special-Auth" not in request.headers:
            cache_key = f"missing_auth_log:{request.remote}"
            if not await self.bot.redis.exists(cache_key):
                log.warning(f"Missing X-Special-Auth header from {request.remote}")
                await self.bot.redis.set(cache_key, "1", ex=60)
            return web.json_response({"error": "Unauthorized"}, status=401)
        
        try:
            auth_header = request.headers.get("X-Special-Auth")
            if not auth_header:
                log.warning("Missing X-Special-Auth header")
                return web.json_response(
                    {"error": "Unauthorized"}, 
                    status=401
                )

            if auth_header != (os.getenv('SPECIAL_AUTH_SECRET') or ''):
                log.warning("Invalid auth token")
                return web.json_response(
                    {"error": "Invalid authentication"}, 
                    status=401
                )

            data = await request.json()
            user_id = data.get("user_id")
            discord_token = data.get("access_token")

            if not user_id or not discord_token:
                return web.json_response(
                    {"error": "Missing user_id or access_token"}, 
                    status=400
                )

            timestamp = int(datetime.now(timezone.utc).timestamp())
            token_data = f"{user_id}-{timestamp}"
            token = hashlib.sha256(
                f"{token_data}-{os.getenv('TOKEN_SECRET') or ''}".encode()
            ).hexdigest()

            await self.bot.db.execute(
                """
                INSERT INTO access_tokens (user_id, token, discord_token, created_at, expires_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP + INTERVAL '14 days')
                ON CONFLICT (user_id) 
                DO UPDATE SET 
                    token = $2,
                    discord_token = $3,
                    created_at = CURRENT_TIMESTAMP,
                    expires_at = CURRENT_TIMESTAMP + INTERVAL '14 days'
                """,
                int(user_id), token, discord_token
            )

            return web.json_response({
                "success": True,
                "token": token,
                "expires_in": 1209600  
            })

        except Exception as e:
            log.error(f"Error in login endpoint: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def verify_token(self, token: str, user_id: int) -> bool:
        """Verify if a token is valid for a user"""
        try:
            result = await self.bot.db.fetchrow(
                """
                SELECT EXISTS(
                    SELECT 1 
                    FROM access_tokens 
                    WHERE token = $1 
                    AND user_id = $2 
                    AND expires_at > CURRENT_TIMESTAMP
                )
                """,
                token, user_id
            )
            return result[0] if result else False
        except Exception as e:
            log.error(f"Error verifying token: {e}")
            return False

    @route("/update/config", ["POST"])
    @ratelimit(5, 60)
    async def update_config(self, request: Request) -> Response:    
        log.info(f"Handling POST request to /update")
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response(
                        {"error": "User not in guild"}, 
                        status=403
                    )
                except discord.HTTPException as e:
                    return web.json_response(
                        {"error": f"Failed to fetch member: {e}"}, 
                        status=500
                    )

            if not member.guild_permissions.manage_guild:
                return web.json_response({"error": "Missing Manage Server permission"}, status=403)

            data = await request.json()
            
            try:
                if "prefix" in data:
                    await self.bot.db.execute(
                        "UPDATE prefix SET prefix = $1 WHERE guild_id = $2",
                        data["prefix"],
                        int(guild_id)
                    )
                    await self.bot.redis.delete(f"prefix:{guild_id}")
            except Exception as e:
                log.error(f"Error updating prefix: {e}")
                raise

            try:
                if "whitelist" in data:
                    await self.bot.db.execute(
                        """
                        INSERT INTO whitelist (guild_id)
                        VALUES ($1)
                        ON CONFLICT (guild_id) DO NOTHING
                        """,
                        int(guild_id)
                    )
                    
                    await self.bot.db.execute(
                        """
                        UPDATE whitelist 
                        SET status = $2, action = $3 
                        WHERE guild_id = $1
                        """,
                        int(guild_id),
                        data["whitelist"]["enabled"],
                        data["whitelist"]["action"]
                    )
            except Exception as e:
                log.error(f"Error updating whitelist: {e}")
                raise

            try:
                if "vanity" in data:
                    vanity_data = data["vanity"]
                    if vanity_data["enabled"]:
                        await self.bot.db.execute(
                            """
                            INSERT INTO vanity (guild_id)
                            VALUES ($1)
                            ON CONFLICT (guild_id) DO NOTHING
                            """,
                            int(guild_id)
                        )
                        
                        await self.bot.db.execute(
                            """
                            UPDATE vanity 
                            SET role_id = $2, channel_id = $3, template = $4
                            WHERE guild_id = $1
                            """,
                            int(guild_id),
                            int(vanity_data["role_id"]) if vanity_data["role_id"] and vanity_data["role_id"] != "None" else None,
                            int(vanity_data["channel_id"]) if vanity_data["channel_id"] and vanity_data["channel_id"] != "None" else None,
                            vanity_data["template"]
                        )
                    else:
                        await self.bot.db.execute(
                            "DELETE FROM vanity WHERE guild_id = $1",
                            int(guild_id)
                        )
            except Exception as e:
                log.error(f"Error updating vanity: {e}")
                raise

            try:
                if "moderation" in data:
                    mod_data = data["moderation"]
                    
                    invoke_messages = mod_data.get("invoke_messages") or mod_data.get("dm_notifications", {}).get("invoke_messages")
                    if invoke_messages:
                        await self.bot.db.execute(
                            """
                            INSERT INTO settings (guild_id)
                            VALUES ($1)
                            ON CONFLICT (guild_id) DO NOTHING
                            """,
                            int(guild_id)
                        )
                        
                        default_invoke_messages = {
                            "kick": "{user.mention} was kicked",
                            "ban": "{user.mention} was banned",
                            "unban": "{user.mention} was unbanned",
                            "timeout": "{user.mention} was timed out",
                            "untimeout": "{user.mention} was untimed out"
                        }
                        
                        for action, msg_data in invoke_messages.items():
                            if msg_data["enabled"]:
                                await self.bot.db.execute(
                                    f"""
                                    UPDATE settings 
                                    SET invoke_{action} = $2
                                    WHERE guild_id = $1
                                    """,
                                    int(guild_id),
                                    msg_data["message"] if msg_data["message"] != default_invoke_messages[action] else None
                                )
                            else:
                                await self.bot.db.execute(
                                    f"""
                                    UPDATE settings 
                                    SET invoke_{action} = NULL
                                    WHERE guild_id = $1
                                    """,
                                    int(guild_id)
                                )
                    
                    if mod_data.get("dm_notifications"):
                        dm_data = mod_data["dm_notifications"]
                        
                        default_messages = {
                                        "ban": "{embed}{title: Banned}{color: #ED4245}{timestamp}$v{field: name: You have been banned in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                                        "kick": "{embed}{title: Kicked}{color: #ED4245}{timestamp}$v{field: name: You have been kicked from && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                                        "mute": "{embed}{title: Muted}{color: #ED4245}{timestamp}$v{field: name: You have been muted in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                                        "unban": "{embed}{title: Unbanned}{color: #57F287}{timestamp}$v{field: name: You have been unbanned from && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                                        "jail": "{embed}{title: Jailed}{color: #ED4245}{timestamp}$v{field: name: You have been jailed in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                                        "unjail": "{embed}{title: Unjailed}{color: #57F287}{timestamp}$v{field: name: You have been unjailed in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}",
                                        "unmute": "{embed}{title: Unmuted}{color: #57F287}{timestamp}$v{field: name: You have been unmuted in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}",
                                        "warn": "{embed}{title: Warned}{color: #ED4245}{timestamp}$v{field: name: You have been warned in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                                        "timeout": "{embed}{title: Timed Out}{description: {duration}}{color: #ED4245}{timestamp}$v{field: name: You have been timed out in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                                        "untimeout": "{embed}{title: Timeout Removed}{color: #57F287}{timestamp}$v{field: name: Your timeout has been removed in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}"
                        }
                        
                        exists = await self.bot.db.fetchval(
                            "SELECT EXISTS(SELECT 1 FROM mod WHERE guild_id = $1)",
                            int(guild_id)
                        )
                        
                        if not exists:
                            await self.bot.db.execute(
                                "INSERT INTO mod (guild_id) VALUES ($1)",
                                int(guild_id)
                            )
                        
                        if "enabled" in dm_data:
                            await self.bot.db.execute(
                                "UPDATE mod SET dm_enabled = $1 WHERE guild_id = $2",
                                bool(dm_data["enabled"]),  
                                int(guild_id)
                            )
                        
                        if "messages" in dm_data:
                            for action, message in dm_data["messages"].items():
                                column = f"dm_{action}"
                                if column in [
                                    "dm_ban", "dm_unban", "dm_kick", "dm_jail", "dm_unjail",
                                    "dm_mute", "dm_unmute", "dm_warn", "dm_timeout", "dm_untimeout"
                                ]:
                                    if message != default_messages.get(action):
                                        await self.bot.db.execute(
                                            f"UPDATE mod SET {column} = $1 WHERE guild_id = $2",
                                            str(message) if message else None,
                                            int(guild_id)
                                        )
                        
                        if "actions" in dm_data:
                            for action, enabled in dm_data["actions"].items():
                                column = f"dm_{action}"
                                if column in [
                                    "dm_antinuke_ban", "dm_antinuke_kick", "dm_antinuke_strip",
                                    "dm_antiraid_ban", "dm_antiraid_kick", "dm_antiraid_timeout", 
                                    "dm_antiraid_strip", "dm_role_add", "dm_role_remove"
                                ]:
                                    await self.bot.db.execute(
                                        f"UPDATE mod SET {column} = $1 WHERE guild_id = $2",
                                        bool(enabled),
                                        int(guild_id)
                                    )
            except Exception as e:
                log.error(f"Error updating moderation settings: {e}")
                raise

            try:
                if "confessions" in data:
                    conf_data = data["confessions"]
                    
                    if conf_data.get("enabled"):
                        await self.bot.db.execute(
                            """
                            DELETE FROM confess 
                            WHERE guild_id = $1
                            """,
                            int(guild_id)
                        )
                        
                        await self.bot.db.execute(
                            """
                            INSERT INTO confess (guild_id, channel_id, confession)
                            VALUES ($1, $2, 0)
                            """,
                            int(guild_id),
                            int(conf_data["channel_id"]) if conf_data.get("channel_id") else None
                        )

                        if "reactions" in conf_data:
                            await self.bot.db.execute(
                                """
                                UPDATE confess 
                                SET upvote = $2, downvote = $3
                                WHERE guild_id = $1
                                """,
                                int(guild_id),
                                conf_data["reactions"].get("upvote", "👍"),
                                conf_data["reactions"].get("downvote", "👎")
                            )

                        if "blacklisted_words" in conf_data:
                            await self.bot.db.execute(
                                "DELETE FROM confess_blacklist WHERE guild_id = $1",
                                int(guild_id)
                            )
                            for word in conf_data["blacklisted_words"]:
                                await self.bot.db.execute(
                                    """
                                    INSERT INTO confess_blacklist (guild_id, word)
                                    VALUES ($1, $2)
                                    """,
                                    int(guild_id), word
                                )

                        if "muted_users" in conf_data:
                            await self.bot.db.execute(
                                "DELETE FROM confess_mute WHERE guild_id = $1",
                                int(guild_id)
                            )
                            for user_id in conf_data["muted_users"]:
                                await self.bot.db.execute(
                                    """
                                    INSERT INTO confess_mute (guild_id, user_id)
                                    VALUES ($1, $2)
                                    """,
                                    int(guild_id), int(user_id)
                                )
                    else:
                        await self.bot.db.execute(
                            "DELETE FROM confess WHERE guild_id = $1",
                            int(guild_id)
                        )
            except Exception as e:
                log.error(f"Error updating confessions: {e}")
                raise

            try:
                if "join_dm" in data:
                    jdm_data = data["join_dm"]
                    
                    if jdm_data.get("enabled"):
                        await self.bot.db.execute(
                            """
                            INSERT INTO joindm.config (guild_id, enabled, message)
                            VALUES ($1, $2, $3)
                            ON CONFLICT (guild_id) 
                            DO UPDATE SET 
                                enabled = $2,
                                message = $3
                            """,
                            int(guild_id),
                            True,
                            jdm_data.get("message")
                        )
                    else:
                        await self.bot.db.execute(
                            """
                            UPDATE joindm.config 
                            SET enabled = false
                            WHERE guild_id = $1
                            """,
                            int(guild_id)
                        )
            except Exception as e:
                log.error(f"Error updating join DM: {e}")
                raise

            try:
                if "restricted_commands" in data:
                    await self.bot.db.execute(
                        """
                        DELETE FROM commands.restricted 
                        WHERE guild_id = $1
                        """,
                        int(guild_id)
                    )
                    
                    for command, role_id in data["restricted_commands"].items():
                        await self.bot.db.execute(
                            """
                            INSERT INTO commands.restricted (guild_id, command, role_id)
                            VALUES ($1, $2, $3)
                            """,
                            int(guild_id),
                            command,
                            int(role_id) if role_id else None
                        )
            except Exception as e:
                log.error(f"Error updating restricted commands: {e}")
                raise

            config_data = await self.fetch_config(int(guild_id))
            return web.json_response(config_data)

        except Exception as e:
            log.error(f"Error updating config: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    async def fetch_config(self, guild_id: int) -> dict:
        """Helper method to fetch config data"""
        prefix = await self.bot.db.fetchval(
            "SELECT prefix FROM prefix WHERE guild_id = $1",
            guild_id
        ) or ";"

        mod_data = await self.bot.db.fetchrow(
            "SELECT * FROM mod WHERE guild_id = $1",
            guild_id
        )
        
        mod_dict = dict(mod_data) if mod_data else {}

        whitelist = await self.bot.db.fetchrow(
            "SELECT status as enabled, action FROM whitelist WHERE guild_id = $1",
            guild_id
        )
        whitelist_dict = dict(whitelist) if whitelist else {"enabled": False, "action": "kick"}

        vanity = await self.bot.db.fetchrow(
            "SELECT role_id, channel_id, template FROM vanity WHERE guild_id = $1",
            guild_id
        )
        vanity_dict = dict(vanity) if vanity else {"role_id": None, "channel_id": None, "template": None}

        settings = await self.bot.db.fetchrow(
            "SELECT invoke_kick, invoke_ban, invoke_unban, invoke_timeout, invoke_untimeout FROM settings WHERE guild_id = $1",
            guild_id
        )
        settings_dict = dict(settings) if settings else {}

        default_invoke_messages = {
            "kick": "{user.mention} was kicked",
            "ban": "{user.mention} was banned",
            "unban": "{user.mention} was unbanned",
            "timeout": "{user.mention} was timed out",
            "untimeout": "{user.mention} was untimed out"
        }

        return {
            "prefix": prefix,
            "moderation": {
                "dm_notifications": {
                    "enabled": mod_dict.get("dm_enabled", False),
                    "actions": {
                        action.replace("dm_", ""): bool(value)  
                        for action, value in mod_dict.items()
                        if action.startswith("dm_") and action != "dm_enabled" 
                        and isinstance(value, bool)  
                    },
                    "messages": {
                        action.replace("dm_", ""): value
                        for action, value in mod_dict.items()
                        if action.startswith("dm_") and action != "dm_enabled" 
                        and isinstance(value, str)
                    }
                },
                "invoke_messages": {
                    action: {
                        "enabled": settings_dict.get(f"invoke_{action}") is not None,
                        "message": settings_dict.get(f"invoke_{action}") or default
                    }
                    for action, default in default_invoke_messages.items()
                }
            },
            "whitelist": whitelist_dict,
            "vanity": {
                "enabled": bool(vanity_dict.get("role_id") or vanity_dict.get("channel_id")),
                "role_id": str(vanity_dict["role_id"]) if vanity_dict.get("role_id") else None,
                "channel_id": str(vanity_dict["channel_id"]) if vanity_dict.get("channel_id") else None,
                "template": vanity_dict.get("template")
            }
        }

    @route("/dashboard/guilds", ["GET"])
    async def get_dashboard_guilds(self, request: Request) -> Response:
        """Get guilds for the dashboard"""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            cache_key = f"dashboard:guilds:{user_data['user_id']}"
            cached_data = await self.bot.redis.get(cache_key)
            
            if cached_data:
                if isinstance(cached_data, bytes):
                    cached_data = cached_data.decode('utf-8')
                if isinstance(cached_data, str):
                    cached_data = json.loads(cached_data)
                cached_data["cached"] = True
                return web.json_response(cached_data)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://discord.com/api/v10/users/@me/guilds",
                        headers={"Authorization": f"Bearer {user_data['discord_token']}"},
                        timeout=5.0
                    ) as resp:
                        if resp.status == 200:
                            user_guilds = await resp.json()
                            
                            bot_guild_ids = {str(g.id) for g in self.bot.guilds}
                            
                            enhanced_guilds = []
                            for guild in user_guilds:
                                permissions = int(guild.get("permissions", 0))
                                guild_object = {
                                    **guild,
                                    "mutual": guild["id"] in bot_guild_ids,
                                    "permissions": {
                                        "admin": (permissions & 0x8) == 0x8,
                                        "manage_guild": (permissions & 0x20) == 0x20,
                                        "manage_roles": (permissions & 0x10000000) == 0x10000000,
                                        "manage_channels": (permissions & 0x10) == 0x10,
                                        "kick_members": (permissions & 0x2) == 0x2,
                                        "ban_members": (permissions & 0x4) == 0x4,
                                        "value": permissions
                                    }
                                }
                                enhanced_guilds.append(guild_object)

                            async with session.get(
                                "https://discord.com/api/v10/users/@me",
                                headers={"Authorization": f"Bearer {user_data['discord_token']}"},
                                timeout=2.0
                            ) as resp:
                                if resp.status == 200:
                                    user_info = await resp.json()
                                else:
                                    raise Exception("Failed to fetch user info")

                            response_data = {
                                "success": True,
                                "cached": False,
                                "user": user_info,
                                "guilds": enhanced_guilds
                            }

                            await self.bot.redis.set(
                                cache_key,
                                json.dumps(response_data),
                                ex=300
                            )
                            
                            return web.json_response(response_data)
                        
                        raise Exception(f"Discord API error: {resp.status}")

            except (asyncio.TimeoutError, Exception) as e:
                log.warning(f"Falling back to mutual guilds due to: {e}")
                
                member = await self.bot.get_or_fetch_member(user_data['user_id'])
                if not member:
                    return web.json_response({"error": "User not found"}, status=404)

                mutual_guilds = []
                for guild in self.bot.guilds:
                    member = guild.get_member(user_data['user_id'])
                    if member:
                        permissions = member.guild_permissions.value
                        mutual_guilds.append({
                            "id": str(guild.id),
                            "name": guild.name,
                            "icon": str(guild.icon) if guild.icon else None,
                            "owner": member.guild.owner_id == member.id,
                            "permissions": {
                                "admin": member.guild_permissions.administrator,
                                "manage_guild": member.guild_permissions.manage_guild,
                                "manage_roles": member.guild_permissions.manage_roles,
                                "manage_channels": member.guild_permissions.manage_channels,
                                "kick_members": member.guild_permissions.kick_members,
                                "ban_members": member.guild_permissions.ban_members,
                                "value": permissions
                            },
                            "mutual": True
                        })

                response_data = {
                    "success": True,
                    "cached": False,
                    "fallback": True,
                    "user": {
                        "id": str(member.id),
                        "username": member.name,
                        "discriminator": member.discriminator,
                        "avatar": str(member.avatar) if member.avatar else None
                    },
                    "guilds": mutual_guilds
                }

                await self.bot.redis.set(
                    cache_key,
                    json.dumps(response_data),
                    ex=60  
                )

                return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error in dashboard guilds endpoint: {e}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/dashboard/user", ["GET"])
    async def get_dashboard_user(self, request: Request) -> Response:
        """Get user dashboard info including instances and donator status"""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bearer {user_data['discord_token']}"}
                ) as resp:
                    if resp.status != 200:
                        return web.json_response(
                            {"error": "Failed to fetch Discord user"}, 
                            status=resp.status
                        )
                    user_info = await resp.json()
            
            instance = await self.bot.db.fetchrow(
                """
                SELECT id, status, expires_at, purchased_at, email
                FROM instances 
                WHERE user_id = $1 
                AND status != 'cancelled'
                ORDER BY purchased_at DESC 
                LIMIT 1
                """,
                user_data['user_id']
            )

            is_donator = await self.bot.db.fetchrow(
                """
                SELECT user_id 
                FROM donators 
                WHERE user_id = $1
                """,
                user_data['user_id']
            )
            
            response_data = {
                "success": True,
                "user": user_info,
                "donator": bool(is_donator),
                "instance": None if not instance else {
                    "id": instance['id'],
                    "status": instance['status'],
                    "expires_at": instance['expires_at'].isoformat() if instance['expires_at'] else None,
                    "purchased_at": instance['purchased_at'].isoformat() if instance['purchased_at'] else None,
                    "email": instance['email']
                }
            }
            
            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error in dashboard user endpoint: {e}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/statistics")
    async def statistics(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            days = int(request.headers.get("X-DAYS", "7"))

            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response(
                        {"error": "User not in guild"}, 
                        status=403
                    )
                except discord.HTTPException as e:
                    return web.json_response(
                        {"error": f"Failed to fetch member: {e}"}, 
                        status=500
                    )

            if not member.guild_permissions.manage_guild:
                return web.json_response(
                    {"error": "Missing Manage Server permission"}, 
                    status=403
                )

            days = int(request.headers.get("X-DAYS", "7"))

            dates = [
                (datetime.now(timezone.utc) - timedelta(days=i)).date()
                for i in range(days)
            ]

            stats_query = f"""
            WITH daily_stats AS (
                SELECT 
                    date_trunc('day', date)::date as day,
                    SUM(messages_sent) as messages,
                    SUM(voice_minutes) as voice_minutes
                FROM statistics.daily 
                WHERE guild_id = $1
                AND date > CURRENT_DATE - INTERVAL '{days} days'
                GROUP BY day
                ORDER BY day
            ),
            filled_days AS (
                SELECT 
                    ad.day,
                    COALESCE(dm.messages, 0) as messages_sent,
                    COALESCE(dm.voice_minutes, 0) as voice_minutes
                FROM (
                    SELECT generate_series(
                        CURRENT_DATE - INTERVAL '{days} days',
                        CURRENT_DATE,
                        '1 day'::interval
                    )::date as day
                ) ad
                LEFT JOIN daily_stats dm ON dm.day = ad.day
                ORDER BY ad.day
            )
            SELECT 
                day::text as date,
                messages_sent,
                voice_minutes
            FROM filled_days
            """

            command_query = f"""
            SELECT 
                DATE(timestamp)::text as date,
                COUNT(*) as commands_used
            FROM invoke_history.commands
            WHERE guild_id = $1
                AND timestamp >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY DATE(timestamp)
            ORDER BY date
            """

            stats_records = await self.bot.db.fetch(stats_query, int(guild_id))
            command_records = await self.bot.db.fetch(command_query, int(guild_id))

            commands_by_date = {r['date']: r['commands_used'] for r in command_records}

            mod_query = """
            SELECT 
                DATE(timestamp)::text as date,
                action,
                COUNT(*) as count
            FROM history.moderation
            WHERE guild_id = $1
                AND timestamp >= $2
                AND timestamp < $2 + INTERVAL '1 day' * $3
            GROUP BY DATE(timestamp), action
            ORDER BY date DESC;
            """

            mod_records = await self.bot.db.fetch(
                mod_query, int(guild_id), dates[-1], days
            )

            mod_by_date = {}
            for record in mod_records:
                if record["date"] not in mod_by_date:
                    mod_by_date[record["date"]] = {}
                mod_by_date[record["date"]][record["action"]] = record["count"]

            response_data = {
                "guild_id": guild_id,
                "days": days,
                "statistics": [
                    {
                        "date": record["date"],
                        "commands_used": commands_by_date.get(record["date"], 0),
                        "messages_sent": record["messages_sent"],
                        "voice_minutes": record["voice_minutes"],
                        "moderation": mod_by_date.get(record["date"], {}),
                    }
                    for record in stats_records
                ],
            }

            return web.json_response(response_data)

        except ValueError as e:
            return web.json_response({"error": "Invalid X-DAYS value"}, status=400)
        except Exception as e:
            log.error(f"Error fetching statistics: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/invokes")
    async def invokes(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response(
                        {"error": "User not in guild"}, 
                        status=403
                    )
                except discord.HTTPException as e:
                    return web.json_response(
                        {"error": f"Failed to fetch member: {e}"}, 
                        status=500
                    )

            if not member.guild_permissions.manage_guild:
                return web.json_response(
                    {"error": "Missing Manage Server permission"}, 
                    status=403
                )

            query = """
            SELECT 
                user_id,
                command_name,
                category,
                timestamp::text
            FROM invoke_history.commands
            WHERE 
                guild_id = $1
                AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '30 days'
            ORDER BY timestamp DESC;
            """

            records = await self.bot.db.fetch(query, int(guild_id))

            unique_users = {record["user_id"] for record in records}
            users_data = []

            for user_id in unique_users:
                user = self.bot.get_user(user_id)
                if user:
                    avatar_url = (
                        str(user.avatar.url)
                        if user.avatar
                        else "https://cdn.discordapp.com/embed/avatars/1.png"
                    )
                    users_data.append(
                        {
                            "user_id": str(user_id),
                            "user_name": user.name,
                            "user_displayname": user.display_name,
                            "user_avatar": avatar_url,
                        }
                    )
                else:
                    users_data.append(
                        {
                            "user_id": str(user_id),
                            "user_name": "Unknown User",
                            "user_displayname": "Unknown User",
                            "user_avatar": "https://cdn.discordapp.com/embed/avatars/1.png",
                        }
                    )

            response_data = {
                "guild_id": guild_id,
                "total_invokes": len(records),
                "users": users_data,
                "invokes": [
                    {
                        "user_id": str(record["user_id"]),
                        "command": record["command_name"],
                        "category": record["category"],
                        "timestamp": record["timestamp"],
                    }
                    for record in records
                ],
            }

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error fetching invoke history: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/security")
    async def security(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response(
                        {"error": "User not in guild"}, 
                        status=403
                    )
                except discord.HTTPException as e:
                    return web.json_response(
                        {"error": f"Failed to fetch member: {e}"}, 
                        status=500
                    )

            has_manage_guild = member.guild_permissions.manage_guild
            has_administrator = member.guild_permissions.administrator
            is_owner = member.id == guild.owner_id
            
            if not (has_manage_guild or has_administrator or is_owner):
                return web.json_response(
                    {"error": "Missing required permissions. Need Manage Server, Administrator, or be the server owner."}, 
                    status=403
                )

            has_manage_guild = member.guild_permissions.manage_guild
            antinuke_settings = await self.bot.db.fetchrow(
                "SELECT * FROM antinuke WHERE guild_id = $1",
                guild.id
            )
            antiraid_settings = await self.bot.db.fetchrow(
                "SELECT * FROM antiraid WHERE guild_id = $1",
                guild.id
            )

            antinuke_dict = dict(antinuke_settings) if antinuke_settings else None
            antiraid_dict = dict(antiraid_settings) if antiraid_settings else None

            if antinuke_dict:
                if 'whitelist' in antinuke_dict:
                    antinuke_dict['whitelist'] = [str(uid) for uid in antinuke_dict['whitelist']]
                if 'trusted_admins' in antinuke_dict:
                    antinuke_dict['trusted_admins'] = [str(uid) for uid in antinuke_dict['trusted_admins']]

                is_trusted = member.id in ({guild.owner_id} | set(self.bot.owner_ids) | set(antinuke_settings['trusted_admins'] if antinuke_settings else []))

            response_data = {
                "guild_id": str(guild.id),
                "permissions": {
                    "manage_guild": has_manage_guild,
                    "trusted_antinuke": is_trusted,
                    "owner": member.id == guild.owner_id
                },
                "antiraid": antiraid_dict if has_manage_guild else None,
                "antinuke": antinuke_dict if is_trusted else None
            }

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error fetching security settings: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/modlogs")
    async def modlogs(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response(
                        {"error": "User not in guild"}, 
                        status=403
                    )
                except discord.HTTPException as e:
                    return web.json_response(
                        {"error": f"Failed to fetch member: {e}"}, 
                        status=500
                    )

            if not member.guild_permissions.administrator and member.id != guild.owner_id:
                return web.json_response(
                    {"error": "Missing Administrator permission"}, 
                    status=403
                )

            mod_config = await self.bot.db.fetchrow(
                "SELECT * FROM mod WHERE guild_id = $1",
                guild.id
            )

            cases = await self.bot.db.fetch(
                """
                SELECT * FROM history.moderation 
                WHERE guild_id = $1 
                ORDER BY timestamp DESC 
                LIMIT 100
                """,
                guild.id
            )

            cases_data = []
            for case in cases:
                case_dict = dict(case)
                case_dict['timestamp'] = case_dict['timestamp'].isoformat() if case_dict['timestamp'] else None
                case_dict['user_id'] = str(case_dict['user_id'])
                case_dict['moderator_id'] = str(case_dict['moderator_id'])
                case_dict['guild_id'] = str(case_dict['guild_id'])
                case_dict['role_id'] = str(case_dict['role_id']) if case_dict.get('role_id') else None
                cases_data.append(case_dict)

            unique_users = {
                int(case['user_id']) for case in cases
            } | {
                int(case['moderator_id']) for case in cases
            }

            users_data = []
            cached_users = {}

            for user_id in unique_users:
                member = guild.get_member(user_id)
                if member:
                    cached_users[user_id] = member

            remaining_ids = unique_users - set(cached_users.keys())
            for user_id in remaining_ids:
                user = self.bot.get_user(user_id)
                if user:
                    cached_users[user_id] = user

            remaining_ids = unique_users - set(cached_users.keys())
            if remaining_ids:
                try:
                    remaining_users = await self.bot.fetch_users(remaining_ids)
                    cached_users.update({user.id: user for user in remaining_users})
                except:
                    pass

            for user_id in unique_users:
                user = cached_users.get(user_id)
                if user:
                    avatar_url = (
                        str(user.display_avatar.url) 
                        if getattr(user, 'display_avatar', None)
                        else "https://cdn.discordapp.com/embed/avatars/1.png"
                    )
                    users_data.append({
                        "user_id": str(user_id),
                        "user_name": user.name,
                        "user_displayname": getattr(user, 'display_name', user.name),
                        "user_avatar": avatar_url,
                    })
                else:
                    users_data.append({
                        "user_id": str(user_id),
                        "user_name": "Unknown User",
                        "user_displayname": "Unknown User",
                        "user_avatar": "https://cdn.discordapp.com/embed/avatars/1.png",
                    })

            response_data = {
                "guild_id": str(guild.id),
                "enabled": bool(mod_config),
                "config": dict(mod_config) if mod_config else None,
                "users": users_data,
                "cases": cases_data
            }

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error fetching modlogs: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/logging")
    async def logging(self: "Network", request: Request) -> Response:
        try:
            print(f"Processing logging request with headers: {dict(request.headers)}")
            
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                log.warning("Missing or invalid authorization header")
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            print(f"Processing request for token: {token[:6]}...")  
            
            try:
                user_data = await self.bot.db.fetchrow(
                    """
                    SELECT user_id, discord_token
                    FROM access_tokens 
                    WHERE token = $1 
                    AND expires_at > CURRENT_TIMESTAMP
                    """,
                    token
                )
                print(f"User data found: {bool(user_data)}")
            except Exception as db_error:
                print(f"Database error: {str(db_error)}\n{traceback.format_exc()}")
                return web.json_response({"error": str(db_error)}, status=500)

            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            try:
                guild = self.bot.get_guild(int(guild_id))
                if not guild:
                    return web.json_response(
                        {"error": "Guild not found"}, 
                        status=404
                    )

                member = guild.get_member(user_data['user_id'])
                if not member:
                    try:
                        member = await guild.fetch_member(user_data['user_id'])
                    except discord.NotFound:
                        return web.json_response(
                            {"error": "User not in guild"}, 
                            status=403
                        )
                    except discord.HTTPException as e:
                        log.error(f"Discord API error fetching member: {e}")
                        return web.json_response(
                            {"error": f"Failed to fetch member: {e}"}, 
                            status=500
                        )

                if not member.guild_permissions.manage_guild:
                    return web.json_response(
                        {"error": "Missing Manage Server permission"}, 
                        status=403
                    )

                try:
                    logging_channels = await self.bot.db.fetch(
                        """
                        SELECT channel_id, events
                        FROM logging
                        WHERE guild_id = $1
                        """,
                        guild.id
                    )
                except Exception as e:
                    print(f"Database error fetching logging channels: {e}")
                    return web.json_response({"error": "Database error"}, status=500)

                log_types = {
                    "MESSAGE": 1,
                    "MEMBER": 2,
                    "MODERATION": 4,
                    "VOICE": 8,
                    "CHANNEL": 16,
                    "ROLE": 32,
                    "INVITE": 64,
                    "EMOJI": 128,
                    "GUILD": 256,
                    "ALL": 511
                }

                channels_data = []
                for record in logging_channels:
                    channel = guild.get_channel(record['channel_id'])
                    if channel:
                        enabled_events = []
                        for event_name, event_value in log_types.items():
                            if record['events'] & event_value:
                                enabled_events.append(event_name)

                        channels_data.append({
                            "channel_id": str(channel.id),
                            "channel_name": channel.name,
                            "events": record['events'],
                            "enabled_events": enabled_events
                        })

                try:
                    logs = await self.bot.db.fetch(
                        """
                        SELECT 
                            id,
                            channel_id,
                            event_type,
                            content,
                            created_at
                        FROM logging_history 
                        WHERE guild_id = $1 
                        ORDER BY created_at DESC 
                        LIMIT 1500
                        """,
                        guild.id
                    )
                except Exception as e:
                    print(f"Database error fetching logs: {e}\n{traceback.format_exc()}")
                    return web.json_response({"error": "Database error"}, status=500)

                logs_data = []
                for log in logs:
                    try:
                        log_entry = {
                            "id": str(log['id']),
                            "channel_id": str(log['channel_id']) if log['channel_id'] else None,
                            "event_type": log['event_type'],
                            "content": json.loads(log['content']) if isinstance(log['content'], str) else log['content'],
                            "created_at": log['created_at'].isoformat()
                        }

                        content = log_entry['content']
                        if (content.get('files') and 
                            len(content['files']) == 1 and 
                            content['files'][0].startswith('messages') and 
                            content['files'][0].endswith('.txt')):
                            
                            if content.get('embed', {}).get('description'):
                                desc = content['embed']['description']
                                if match := re.search(r'(\d+) messages', desc):
                                    num_messages = int(match.group(1))
                                    log_entry['content']['bulk_deleted_messages'] = {
                                        'count': num_messages,
                                        'channel_id': content.get('target_channel_id')
                                    }

                        logs_data.append(log_entry)
                    except Exception as e:
                        log.error(f"Error processing log entry: {e}")
                        continue

                response_data = {
                    "guild_id": str(guild.id),
                    "enabled": bool(channels_data),
                    "channels": channels_data,
                    "available_events": log_types,
                    "logs": logs_data
                }

                return web.json_response(response_data)

            except Exception as e:
                print(f"Error processing guild data: {e}\n{traceback.format_exc()}")
                return web.json_response({"error": "Internal server error"}, status=500)

        except Exception as e:
            print(f"Logging endpoint error: {str(e)}\n{traceback.format_exc()}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/config")
    async def config(self: "Network", request: Request) -> Response:
        DEFAULT_PREFIX = ";"
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response({"error": "Missing authorization"}, status=401)
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response({"error": "Invalid or expired token"}, status=401)

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response({"error": "Missing X-GUILD-ID header"}, status=400)

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response({"error": "Guild not found"}, status=404)

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response({"error": "User not in guild"}, status=403)
                except discord.HTTPException as e:
                    return web.json_response({"error": f"Failed to fetch member: {e}"}, status=500)

            if not member.guild_permissions.manage_guild:
                return web.json_response({"error": "Missing Manage Server permission"}, status=403)

            prefix_data = await self.bot.redis.get(f"prefix:{guild_id}")
            if not prefix_data:
                prefix_data = await self.bot.db.fetchval(
                    "SELECT prefix FROM prefix WHERE guild_id = $1",
                    int(guild_id)
                )

            mod_settings = await self.bot.db.fetchrow(
                "SELECT * FROM mod WHERE guild_id = $1",
                int(guild_id)
            )

            poj_channels = await self.bot.db.fetch(
                "SELECT channel_id FROM pingonjoin WHERE guild_id = $1",
                int(guild_id)
            )

            settings = await self.bot.db.fetchrow(
                """
                SELECT 
                    invoke_kick,
                    invoke_ban,
                    invoke_unban,
                    invoke_timeout,
                    invoke_untimeout
                FROM settings 
                WHERE guild_id = $1
                """,
                int(guild_id)
            )

            whitelist_settings = await self.bot.db.fetchrow(
                """
                SELECT status, action
                FROM whitelist
                WHERE guild_id = $1
                """,
                int(guild_id)
            )

            vanity_settings = await self.bot.db.fetchrow(
                """
                SELECT role_id, channel_id, template
                FROM vanity
                WHERE guild_id = $1
                """,
                int(guild_id)
            )

            confessions_blacklist = await self.bot.db.fetch(
                """
                SELECT word 
                FROM confess_blacklist 
                WHERE guild_id = $1
                ORDER BY word
                """,
                int(guild_id)
            )

            confessions_muted = await self.bot.db.fetch(
                """
                SELECT user_id 
                FROM confess_mute 
                WHERE guild_id = $1
                """,
                int(guild_id)
            )

            confessions_config = await self.bot.db.fetchrow(
                """
                SELECT channel_id, upvote, downvote, confession
                FROM confess 
                WHERE guild_id = $1
                """,
                int(guild_id)
            )

            joindm_config = await self.bot.db.fetchrow(
                """
                SELECT enabled, message
                FROM joindm.config
                WHERE guild_id = $1
                """,
                int(guild_id)
            )

            restricted_commands = await self.bot.db.fetch(
                """
                SELECT command, role_id
                FROM commands.restricted
                WHERE guild_id = $1
                """,
                int(guild_id)
            )

            default_invoke_messages = {
                "kick": "{user.mention} was kicked",
                "ban": "{user.mention} was banned",
                "unban": "{user.mention} was unbanned",
                "timeout": "{user.mention} was timed out",
                "untimeout": "{user.mention} was untimed out"
            }

            default_messages = {
                "ban": "{embed}{title: Banned}{color: #ED4245}{timestamp}$v{field: name: You have been banned in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                "kick": "{embed}{title: Kicked}{color: #ED4245}{timestamp}$v{field: name: You have been kicked from && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                "mute": "{embed}{title: Muted}{color: #ED4245}{timestamp}$v{field: name: You have been muted in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                "unban": "{embed}{title: Unbanned}{color: #57F287}{timestamp}$v{field: name: You have been unbanned from && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                "jail": "{embed}{title: Jailed}{color: #ED4245}{timestamp}$v{field: name: You have been jailed in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                "unjail": "{embed}{title: Unjailed}{color: #57F287}{timestamp}$v{field: name: You have been unjailed in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}",
                "unmute": "{embed}{title: Unmuted}{color: #57F287}{timestamp}$v{field: name: You have been unmuted in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}",
                "warn": "{embed}{title: Warned}{color: #ED4245}{timestamp}$v{field: name: You have been warned in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                "timeout": "{embed}{title: Timed Out}{description: {duration}}{color: #ED4245}{timestamp}$v{field: name: You have been timed out in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}$v{field: name: Reason && value: {reason} && inline}",
                "untimeout": "{embed}{title: Timeout Removed}{color: #57F287}{timestamp}$v{field: name: Your timeout has been removed in && value: {guild.name} && inline}$v{field: name: Moderator && value: {moderator} && inline}"
            }

            response_data = {
                "guild_id": guild_id,
                "prefix": prefix_data or DEFAULT_PREFIX,
                "moderation": {
                    "enabled": bool(mod_settings),
                    "dm_notifications": {
                        "enabled": mod_settings["dm_enabled"] if mod_settings else False,
                        "actions": {
                            "antinuke_ban": bool(mod_settings.get("dm_antinuke_ban", False)),
                            "antinuke_kick": bool(mod_settings.get("dm_antinuke_kick", False)),
                            "antinuke_strip": bool(mod_settings.get("dm_antinuke_strip", False)),
                            "antiraid_ban": bool(mod_settings.get("dm_antiraid_ban", False)),
                            "antiraid_kick": bool(mod_settings.get("dm_antiraid_kick", False)),
                            "antiraid_timeout": bool(mod_settings.get("dm_antiraid_timeout", False)),
                            "antiraid_strip": bool(mod_settings.get("dm_antiraid_strip", False)),
                            "role_add": bool(mod_settings.get("dm_role_add", False)),
                            "role_remove": bool(mod_settings.get("dm_role_remove", False)),
                            
                            "ban": bool(mod_settings.get("dm_ban") and not mod_settings["dm_ban"].startswith("{")),
                            "kick": bool(mod_settings.get("dm_kick") and not mod_settings["dm_kick"].startswith("{")),
                            "mute": bool(mod_settings.get("dm_mute") and not mod_settings["dm_mute"].startswith("{")),
                            "unban": bool(mod_settings.get("dm_unban") and not mod_settings["dm_unban"].startswith("{")),
                            "jail": bool(mod_settings.get("dm_jail") and not mod_settings["dm_jail"].startswith("{")),
                            "unjail": bool(mod_settings.get("dm_unjail") and not mod_settings["dm_unjail"].startswith("{")),
                            "unmute": bool(mod_settings.get("dm_unmute") and not mod_settings["dm_unmute"].startswith("{")),
                            "warn": bool(mod_settings.get("dm_warn") and not mod_settings["dm_warn"].startswith("{")),
                            "timeout": bool(mod_settings.get("dm_timeout") and not mod_settings["dm_timeout"].startswith("{")),
                            "untimeout": bool(mod_settings.get("dm_untimeout") and not mod_settings["dm_untimeout"].startswith("{"))
                        },
                        "messages": {
                            action: value if value and value.startswith("{") else default
                            for action, default in default_messages.items()
                            if mod_settings and (value := mod_settings.get(f"dm_{action}"))
                        },
                        "invoke_messages": {
                            action: {
                                "enabled": settings.get(f"invoke_{action}") is not None,
                                "message": settings.get(f"invoke_{action}") or default
                            }
                            for action, default in default_invoke_messages.items()
                        }
                    } if mod_settings else None,
                    "whitelist": {
                        "enabled": whitelist_settings["status"] if whitelist_settings else False,
                        "action": whitelist_settings["action"] if whitelist_settings else "kick"
                    },
                    "vanity": {
                        "enabled": bool(vanity_settings),
                        "role_id": str(vanity_settings["role_id"]) if vanity_settings else None,
                        "channel_id": str(vanity_settings["channel_id"]) if vanity_settings else None,
                        "template": vanity_settings["template"] if vanity_settings else None,
                        "default_template": "{title: vanity set}{description: thank you {user.mention}}{footer: put /{vanity} in your status for the role.}"
                    },
                    "ping_on_join": {
                        "enabled": bool(poj_channels),
                        "channels": [str(record['channel_id']) for record in poj_channels]
                    } if poj_channels else None
                },
                "confessions": {
                    "enabled": bool(confessions_config),
                    "channel_id": str(confessions_config['channel_id']) if confessions_config else None,
                    "total_confessions": confessions_config['confession'] if confessions_config else 0,
                    "reactions": {
                        "upvote": confessions_config['upvote'] if confessions_config else "👍",
                        "downvote": confessions_config['downvote'] if confessions_config else "👎"
                    } if confessions_config else None,
                    "blacklisted_words": [record['word'] for record in confessions_blacklist],
                    "muted_users": [str(record['user_id']) for record in confessions_muted]
                },
                "join_dm": {
                    "enabled": bool(joindm_config and joindm_config['enabled']),
                    "message": joindm_config['message'] if joindm_config else None
                },
                "restricted_commands": {
                    str(record['command']): str(record['role_id'])
                    for record in restricted_commands
                } if restricted_commands else {}
            }

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error fetching config: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/roles")
    async def roles(self: "Network", request: Request) -> Response:
        try:
            # Auth check
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response({"error": "Missing authorization"}, status=401)
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response({"error": "Invalid or expired token"}, status=401)

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response({"error": "Missing X-GUILD-ID header"}, status=400)

            cache_key = f"guild:roles:{guild_id}"
            cached_data = await self.bot.redis.get(cache_key)
            
            if cached_data:
                try:
                    if isinstance(cached_data, bytes):
                        cached_data = cached_data.decode('utf-8')
                    
                    data = json.loads(cached_data) if isinstance(cached_data, str) else cached_data
                    data["cached"] = True
                    return web.json_response(data)
                        
                except Exception as e:
                    log.warning(f"Cache corruption for {cache_key}: {e}")
                    await self.bot.redis.delete(cache_key)

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response({"error": "Guild not found"}, status=404)

            roles_data = []
            for role in sorted(guild.roles, key=lambda r: r.position, reverse=True):
                if role.is_default():
                    continue
                roles_data.append({
                    "id": str(role.id),
                    "name": role.name,
                    "color": role.color.value,
                    "position": role.position,
                    "permissions": role.permissions.value,
                    "mentionable": role.mentionable,
                    "hoist": role.hoist,
                    "managed": role.managed,
                    "icon": str(role.icon.url) if role.icon else None,
                })

            response_data = {
                "roles": roles_data,
                "cached": False
            }

            try:
                cache_data = json.dumps(response_data, ensure_ascii=False)
                await self.bot.redis.set(
                    cache_key,
                    cache_data,
                    ex=120
                )
            except Exception as e:
                log.error(f"Failed to cache roles data: {e}")

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error fetching roles: {e}", exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/channels")
    async def channels(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response({"error": "Missing authorization"}, status=401)
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response({"error": "Invalid or expired token"}, status=401)

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response({"error": "Missing X-GUILD-ID header"}, status=400)

            cache_key = f"guild:channels:{guild_id}"
            cached_data = await self.bot.redis.get(cache_key)
            
            if cached_data:
                try:
                    if isinstance(cached_data, bytes):
                        cached_data = cached_data.decode('utf-8')
                    
                    data = json.loads(cached_data) if isinstance(cached_data, str) else cached_data
                    data["cached"] = True
                    return web.json_response(data)
                        
                except Exception as e:
                    log.warning(f"Cache corruption for {cache_key}: {e}")
                    await self.bot.redis.delete(cache_key)

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response({"error": "Guild not found"}, status=404)

            categories_data = []
            channels_data = []

            for channel in guild.channels:
                if isinstance(channel, discord.CategoryChannel):
                    categories_data.append({
                        "id": str(channel.id),
                        "name": channel.name,
                        "position": channel.position,
                        "nsfw": channel.nsfw
                    })
                else:
                    channel_data = {
                        "id": str(channel.id),
                        "name": channel.name,
                        "position": channel.position,
                        "type": str(channel.type),
                        "category_id": str(channel.category_id) if channel.category_id else None,
                        "nsfw": getattr(channel, 'nsfw', False),
                    }
                    
                    if isinstance(channel, discord.TextChannel):
                        channel_data.update({
                            "topic": channel.topic,
                            "slowmode_delay": channel.slowmode_delay,
                            "news": channel.is_news()
                        })
                    elif isinstance(channel, discord.VoiceChannel):
                        channel_data.update({
                            "bitrate": channel.bitrate,
                            "user_limit": channel.user_limit,
                            "rtc_region": str(channel.rtc_region) if channel.rtc_region else None
                        })
                    elif isinstance(channel, discord.ForumChannel):
                        channel_data.update({
                            "topic": channel.topic,
                            "slowmode_delay": channel.slowmode_delay,
                            "default_auto_archive_duration": channel.default_auto_archive_duration
                        })
                        
                    channels_data.append(channel_data)

            response_data = {
                "categories": categories_data,
                "channels": channels_data,
                "cached": False
            }

            try:
                cache_data = json.dumps(response_data, ensure_ascii=False)
                await self.bot.redis.set(
                    cache_key,
                    cache_data,
                    ex=120
                )
            except Exception as e:
                log.error(f"Failed to cache channels data: {e}")

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error fetching channels: {e}", exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/update/security", ["POST"])
    async def update_security(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response(
                        {"error": "User not in guild"}, 
                        status=403
                    )
                except discord.HTTPException as e:
                    return web.json_response(
                        {"error": f"Failed to fetch member: {e}"}, 
                        status=500
                    )

            has_manage_guild = member.guild_permissions.manage_guild
            antinuke_settings = await self.bot.db.fetchrow(
                "SELECT * FROM antinuke WHERE guild_id = $1",
                guild.id
            )

            is_trusted = member.id in (
                {guild.owner_id} |
                set(antinuke_settings['trusted_admins'] if antinuke_settings else []) |
                set(self.bot.owner_ids)
            )

            data = await request.json()
            
            if "antiraid" in data and has_manage_guild:
                await self.bot.db.execute(
                    """
                    INSERT INTO antiraid (guild_id, locked, joins, mentions, avatar, browser)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (guild_id) 
                    DO UPDATE SET 
                        locked = $2,
                        joins = $3,
                        mentions = $4,
                        avatar = $5,
                        browser = $6
                    """,
                    int(guild_id),
                    data["antiraid"]["locked"],
                    data["antiraid"]["joins"],
                    data["antiraid"]["mentions"],
                    data["antiraid"]["avatar"],
                    data["antiraid"]["browser"]
                )

            if "antinuke" in data and is_trusted:
                antinuke_exists = await self.bot.db.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM antinuke WHERE guild_id = $1)",
                    int(guild_id)
                )
                
                if not antinuke_exists:
                    await self.bot.db.execute(
                        """
                        INSERT INTO antinuke (
                            guild_id, whitelist, trusted_admins, 
                            bot, ban, kick, role, channel, webhook, emoji
                        ) VALUES (
                            $1, $2, $3, 
                            false, $4, $4, $4, $4, $4, $4
                        )
                        """,
                        int(guild_id),
                        [], 
                        [member.id] if member.id == guild.owner_id else [],
                        json.dumps({
                            "threshold": 5,
                            "duration": 60,
                            "punishment": "ban"
                        })
                    )

                current_settings = await self.bot.db.fetchrow(
                    "SELECT whitelist FROM antinuke WHERE guild_id = $1",
                    int(guild_id)
                )
                current_whitelist = current_settings['whitelist'] if current_settings else []
                current_set = {str(x) for x in current_whitelist}
                new_set = set(data['antinuke'].get('whitelist', []))
                whitelist_modified = ("whitelist" in data["antinuke"] and current_set != new_set)

                if whitelist_modified and member.id != guild.owner_id:
                    return web.json_response(
                        {"error": "Only the server owner can modify the antinuke whitelist"}, 
                        status=403
                    )

                whitelist_to_save = [int(x) for x in data["antinuke"]["whitelist"]] if whitelist_modified else current_whitelist

                modules = ['ban', 'kick', 'role', 'channel', 'webhook', 'emoji']
                module_settings = {}
                
                for module in modules:
                    module_data = data["antinuke"].get(module)
                    if module_data and isinstance(module_data, bool) and not module_data:
                        module_settings[module] = False
                    elif module_data:
                        if isinstance(module_data, dict):
                            settings = {
                                "threshold": int(module_data.get("threshold", 5)),
                                "duration": int(module_data.get("duration", 60)),
                                "punishment": str(module_data.get("punishment", "ban"))
                            }
                        else:
                            settings = {
                                "threshold": 5,
                                "duration": 60,
                                "punishment": "ban"
                            }
                        module_settings[module] = json.dumps(settings)
                    else:
                        module_settings[module] = False

                async with self.bot.db.acquire() as conn:
                    async with conn.transaction():
                        await conn.execute(
                            """
                            UPDATE antinuke SET 
                                whitelist = $2,
                                trusted_admins = $3,
                                bot = $4,
                                ban = $5,
                                kick = $6,
                                role = $7,
                                channel = $8,
                                webhook = $9,
                                emoji = $10
                            WHERE guild_id = $1
                            """,
                            int(guild_id),
                            whitelist_to_save,
                            [int(x) for x in data["antinuke"]["trusted_admins"]],
                            data["antinuke"]["bot"],
                            module_settings['ban'],
                            module_settings['kick'],
                            module_settings['role'],
                            module_settings['channel'],
                            module_settings['webhook'],
                            module_settings['emoji']
                        )

            return web.json_response({
                "guild_id": str(guild_id),
                "permissions": {
                    "manage_guild": has_manage_guild,
                    "trusted_antinuke": is_trusted,
                    "owner": member.id == guild.owner_id
                },
                "antiraid": dict(await self.bot.db.fetchrow(
                    "SELECT guild_id, locked, joins, mentions, avatar, browser FROM antiraid WHERE guild_id = $1",
                    int(guild_id)
                )) if has_manage_guild else None,
                "antinuke": dict(await self.bot.db.fetchrow(
                    "SELECT guild_id, whitelist, trusted_admins, bot, ban, kick, role, channel, webhook, emoji FROM antinuke WHERE guild_id = $1",
                    int(guild_id)
                )) if is_trusted else None
            })

        except Exception as e:
            log.error(f"Error updating security settings: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/beta")
    async def beta_check(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            beta_access = await self.bot.db.fetchrow(
                """
                SELECT user_id
                FROM beta_dashboard
                WHERE user_id = $1
                """,
                user_data['user_id']
            )

            return web.json_response({
                "has_access": bool(beta_access)
            })

        except Exception as e:
            log.error(f"Error checking beta access: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/beta/submit", ["POST"])
    async def beta_submit(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                log.error("Missing authorization header")
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                log.error(f"Invalid or expired token: {token}")
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            data = await request.json()
            
            if 'email' not in data or not data['email']:
                data['email'] = 'example@email.com'
                log.info(f"No email provided for user {user_data['user_id']}, using default")
            
            required_fields = ["display_name", "description"]  
            for field in required_fields:
                if field not in data:
                    log.error(f"Missing required field: {field}")
                    return web.json_response(
                        {"error": f"Missing required field: {field}"}, 
                        status=400
                    )
                if not isinstance(data[field], str):
                    log.error(f"Field {field} must be a string, got {type(data[field])}")
                    return web.json_response(
                        {"error": f"Field {field} must be a string"}, 
                        status=400
                    )

            existing = await self.bot.db.fetchrow(
                """
                SELECT status FROM beta_dashboard 
                WHERE user_id = $1
                """,
                user_data['user_id']
            )
            
            if existing:
                log.error(f"User {user_data['user_id']} already has a {existing['status']} beta request")
                return web.json_response(
                    {"error": f"Already have a {existing['status']} beta request"}, 
                    status=400
                )

            await self.bot.db.execute(
                """
                INSERT INTO beta_dashboard (user_id, status, notes)
                VALUES ($1, 'pending', $2)
                """,
                user_data['user_id'],
                f"Email: {data['email']}\nDisplay Name: {data['display_name']}\n\n{data['description']}"
            )

            channel = self.bot.get_channel(1324832222397726803)
            if channel:
                user = self.bot.get_user(user_data['user_id'])
                embed = discord.Embed(
                    title="New Beta Dashboard Request",
                    color=discord.Color.blue(),
                    timestamp=discord.utils.utcnow()
                )
                embed.add_field(name="User", value=f"{user.mention if user else user_data['user_id']}", inline=True)
                embed.add_field(name="Display Name", value=data["display_name"], inline=True)
                embed.add_field(name="Email", value=data["email"], inline=True)
                embed.add_field(name="Description", value=data["description"], inline=False)
                await channel.send(embed=embed)

            return web.json_response({"success": True})

        except Exception as e:
            log.error(f"Error submitting beta request: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/automation")
    async def automation(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response(
                        {"error": "User not in guild"}, 
                        status=403
                    )
                except discord.HTTPException as e:
                    return web.json_response(
                        {"error": f"Failed to fetch member: {e}"}, 
                        status=500
                    )

            if not member.guild_permissions.manage_messages:
                return web.json_response(
                    {"error": "Missing Manage Messages permission"}, 
                    status=403
                )

            profiles = await self.bot.db.fetch(
                """
                SELECT channel_id, type, category
                FROM auto.media
                WHERE guild_id = $1
                ORDER BY channel_id, type
                """,
                guild.id
            )

            profiles_data = []
            for profile in profiles:
                channel = guild.get_channel(profile['channel_id'])
                if channel:
                    profiles_data.append({
                        "channel": {
                            "id": str(channel.id),
                            "name": channel.name,
                            "type": channel.type.name
                        },
                        "type": profile['type'],
                        "category": profile['category']
                    })

            tags = await self.bot.db.fetch(
                """
                SELECT t.name, t.template, t.owner_id, t.uses, t.created_at::text,
                    array_remove(array_agg(ta.alias), NULL) as aliases
                FROM tags t
                LEFT JOIN tag_aliases ta ON t.guild_id = ta.guild_id 
                    AND LOWER(t.name) = LOWER(ta.original)
                WHERE t.guild_id = $1
                GROUP BY t.name, t.template, t.owner_id, t.uses, t.created_at
                ORDER BY t.uses DESC
                """,
                guild.id
            )

            responses = await self.bot.db.fetch(
                """
                SELECT trigger, template, strict, reply, delete, 
                    delete_after, role_id
                FROM response_trigger
                WHERE guild_id = $1
                ORDER BY trigger ASC
                """,
                guild.id
            )

            reactions = await self.bot.db.fetch(
                """
                SELECT trigger, ARRAY_AGG(emoji) as emojis
                FROM reaction_trigger
                WHERE guild_id = $1
                GROUP BY trigger
                ORDER BY trigger ASC
                """,
                guild.id
            )

            tracker = await self.bot.db.fetchrow(
                """
                SELECT vanity_channel_id, username_channel_id
                FROM tracker
                WHERE guild_id = $1
                """,
                guild.id
            )

            unique_users = {record['owner_id'] for record in tags}
            users_data = []

            for user_id in unique_users:
                user = guild.get_member(user_id)
                if not user:
                    user = self.bot.get_user(user_id)
                
                if user:
                    avatar_url = str(user.display_avatar.url) if getattr(user, 'display_avatar', None) else "https://cdn.discordapp.com/embed/avatars/1.png"
                    users_data.append({
                        "id": str(user_id),
                        "name": user.name,
                        "display_name": getattr(user, 'display_name', user.name),
                        "avatar": avatar_url,
                    })
                else:
                    users_data.append({
                        "id": str(user_id),
                        "name": "Unknown User",
                        "display_name": "Unknown User",
                        "avatar": "https://cdn.discordapp.com/embed/avatars/1.png",
                    })

            tags_data = [{
                "name": tag['name'],
                "content": tag['template'],
                "owner_id": str(tag['owner_id']),
                "uses": tag['uses'],
                "created_at": tag['created_at'],
                "aliases": tag['aliases'] if tag['aliases'] else []
            } for tag in tags]

            responses_data = []
            for response in responses:
                response_dict = {
                    "trigger": response['trigger'],
                    "content": response['template'],
                    "settings": {
                        "strict": response['strict'],
                        "reply": response['reply'],
                        "delete": response['delete'],
                        "delete_after": response['delete_after']
                    }
                }
                
                if response['role_id']:
                    role = guild.get_role(response['role_id'])
                    if role:
                        response_dict["role"] = {
                            "id": str(role.id),
                            "name": role.name,
                            "color": role.color.value if role.color else None
                        }

                responses_data.append(response_dict)

            reactions_data = [{
                "trigger": reaction['trigger'],
                "emojis": reaction['emojis']
            } for reaction in reactions]

            tracker_data = {
                "vanity": None,
                "usernames": None
            }

            if tracker:
                if tracker['vanity_channel_id']:
                    channel = guild.get_channel(tracker['vanity_channel_id'])
                    if channel:
                        tracker_data["vanity"] = {
                            "channel": {
                                "id": str(channel.id),
                                "name": channel.name,
                                "type": channel.type.name
                            }
                        }

                if tracker['username_channel_id']:
                    channel = guild.get_channel(tracker['username_channel_id'])
                    if channel:
                        tracker_data["usernames"] = {
                            "channel": {
                                "id": str(channel.id),
                                "name": channel.name,
                                "type": channel.type.name
                            }
                        }

            response_data = {
                "guild_id": str(guild_id),
                "permissions": {
                    "manage_messages": member.guild_permissions.manage_messages,
                    "administrator": member.guild_permissions.administrator,
                    "manage_guild": member.guild_permissions.manage_guild,
                    "manage_channels": member.guild_permissions.manage_channels
                },
                "users": users_data,
                "tags": {
                    "count": len(tags_data),
                    "items": tags_data
                },
                "autoresponses": {
                    "count": len(responses_data),
                    "items": responses_data
                },
                "reactions": {
                    "count": len(reactions_data),
                    "items": reactions_data,
                    "limits": {
                        "max_per_trigger": 3,
                        "trigger_length": 50
                    }
                },
                "trackers": tracker_data,
                "profiles": {
                    "count": len(profiles_data),
                    "items": profiles_data,
                    "types": {
                        "pfp": {
                            "categories": ["random", "anime", "cats", "egirls", "girls", "boys"]
                        },
                        "banner": {
                            "categories": ["random", "anime", "cute", "imsg", "mix"],
                            "case_map": {
                                "anime": "Anime",
                                "cute": "Cute",
                                "imsg": "Imsg",
                                "mix": "Mix",
                                "random": "random"
                            }
                        }
                    }
                }
            }

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error fetching automation data: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/update/automation", ["POST"])
    async def update_automation(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response({"error": "Missing authorization"}, status=401)
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                """
                SELECT user_id, discord_token
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )
            
            if not user_data:
                return web.json_response({"error": "Invalid or expired token"}, status=401)

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response({"error": "Missing X-GUILD-ID header"}, status=400)

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response({"error": "Guild not found"}, status=404)

            member = guild.get_member(user_data['user_id'])
            if not member:
                try:
                    member = await guild.fetch_member(user_data['user_id'])
                except discord.NotFound:
                    return web.json_response({"error": "User not in guild"}, status=403)
                except discord.HTTPException as e:
                    return web.json_response({"error": f"Failed to fetch member: {e}"}, status=500)

            if not member.guild_permissions.manage_messages:
                return web.json_response({"error": "Missing Manage Messages permission"}, status=403)

            data = await request.json()

            try:
                if "tags" in data:
                    for action in data["tags"]:
                        if action["type"] == "create":
                            if not action.get("content") or not action.get("content").strip():
                                return web.json_response({"error": "Tag content cannot be empty"}, status=400)
                            await self.bot.db.execute(
                                """
                                INSERT INTO tags (guild_id, name, owner_id, template)
                                VALUES ($1, $2, $3, $4)
                                """,
                                int(guild_id), action["name"], member.id, action["content"]
                            )
                        elif action["type"] == "edit":
                            if not action.get("content") or not action.get("content").strip():
                                return web.json_response({"error": "Tag content cannot be empty"}, status=400)
                            result = await self.bot.db.execute(
                                """
                                UPDATE tags 
                                SET template = $3
                                WHERE guild_id = $1 AND LOWER(name) = LOWER($2)
                                """,
                                int(guild_id), action["name"], action["content"]
                            )
                            if result == "UPDATE 0":
                                return web.json_response(
                                    {"error": f"Tag '{action['name']}' not found"}, 
                                    status=404
                                )
                        elif action["type"] == "delete":
                            await self.bot.db.execute(
                                """
                                DELETE FROM tags 
                                WHERE guild_id = $1 AND LOWER(name) = LOWER($2)
                                """,
                                int(guild_id), action["name"]
                            )

                if "autoresponses" in data:
                    for action in data["autoresponses"]:
                        if action["type"] == "create":
                            role_id = int(action["settings"].get("role", {}).get("id")) if action["settings"].get("role") else None
                            await self.bot.db.execute(
                                """
                                INSERT INTO response_trigger (
                                    guild_id, trigger, template, strict, reply, 
                                    delete, delete_after, role_id
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                                """,
                                int(guild_id), action["trigger"], action["content"],
                                action["settings"]["strict"], action["settings"]["reply"],
                                action["settings"]["delete"], action["settings"]["delete_after"],
                                role_id
                            )
                        elif action["type"] == "edit":
                            role_id = int(action["settings"].get("role", {}).get("id")) if action["settings"].get("role") else None
                            result = await self.bot.db.execute(
                                """
                                UPDATE response_trigger 
                                SET trigger = $3, template = $4, strict = $5, reply = $6,
                                    delete = $7, delete_after = $8, role_id = $9
                                WHERE guild_id = $1 AND LOWER(trigger) = LOWER($2)
                                """,
                                int(guild_id), 
                                action["original_trigger"], 
                                action["trigger"],         
                                action["content"],
                                action["settings"]["strict"], 
                                action["settings"]["reply"],
                                action["settings"]["delete"], 
                                action["settings"]["delete_after"],
                                role_id
                            )
                            if result == "UPDATE 0":
                                return web.json_response(
                                    {"error": f"Autoresponse trigger '{action['original_trigger']}' not found"}, 
                                    status=404
                                )
                        elif action["type"] == "delete":
                            result = await self.bot.db.execute(
                                """
                                DELETE FROM response_trigger 
                                WHERE guild_id = $1 AND LOWER(trigger) = LOWER($2)
                                """,
                                int(guild_id), action["trigger"]
                            )
                            if result == "DELETE 0":
                                return web.json_response(
                                    {"error": f"Autoresponse trigger '{action['trigger']}' not found"}, 
                                    status=404
                                )

                if "reactions" in data:
                    for action in data["reactions"]:
                        if action["type"] == "create":
                            for emoji in action["emojis"]:
                                await self.bot.db.execute(
                                    """
                                    INSERT INTO reaction_trigger (guild_id, trigger, emoji)
                                    VALUES ($1, LOWER($2), $3)
                                    """,
                                    int(guild_id), action["trigger"], emoji
                                )
                        elif action["type"] == "edit":
                            await self.bot.db.execute(
                                """
                                DELETE FROM reaction_trigger 
                                WHERE guild_id = $1 AND LOWER(trigger) = LOWER($2)
                                """,
                                int(guild_id), action["original_trigger"]
                            )
                            for emoji in action["emojis"]:
                                await self.bot.db.execute(
                                    """
                                    INSERT INTO reaction_trigger (guild_id, trigger, emoji)
                                    VALUES ($1, LOWER($2), $3)
                                    """,
                                    int(guild_id), action["trigger"], emoji
                                )
                        elif action["type"] == "delete":
                            await self.bot.db.execute(
                                """
                                DELETE FROM reaction_trigger 
                                WHERE guild_id = $1 AND LOWER(trigger) = LOWER($2)
                                """,
                                int(guild_id), action["trigger"]
                            )

                if "profiles" in data:
                    for action in data["profiles"]:
                        if action["type"] == "create":
                            await self.bot.db.execute(
                                """
                                INSERT INTO auto.media (guild_id, channel_id, type, category)
                                VALUES ($1, $2, $3, $4)
                                ON CONFLICT (guild_id, channel_id, type)
                                DO UPDATE SET category = EXCLUDED.category
                                """,
                                int(guild_id), int(action["channel"]["id"]),
                                action["media_type"], action["category"]
                            )
                        elif action["type"] == "delete":
                            await self.bot.db.execute(
                                """
                                DELETE FROM auto.media
                                WHERE guild_id = $1 AND channel_id = $2 AND type = $3
                                """,
                                int(guild_id), int(action["channel"]["id"]), action["media_type"]
                            )

                if "trackers" in data:
                    if "vanity" in data["trackers"]:
                        channel_id = int(data["trackers"]["vanity"]["channel"]["id"]) if data["trackers"]["vanity"] else None
                        await self.bot.db.execute(
                            """
                            INSERT INTO tracker (guild_id, vanity_channel_id)
                            VALUES ($1, $2)
                            ON CONFLICT (guild_id)
                            DO UPDATE SET vanity_channel_id = EXCLUDED.vanity_channel_id
                            """,
                            int(guild_id), channel_id
                        )

                    if "usernames" in data["trackers"]:
                        channel_id = int(data["trackers"]["usernames"]["channel"]["id"]) if data["trackers"]["usernames"] else None
                        await self.bot.db.execute(
                            """
                            INSERT INTO tracker (guild_id, username_channel_id)
                            VALUES ($1, $2)
                            ON CONFLICT (guild_id)
                            DO UPDATE SET username_channel_id = EXCLUDED.username_channel_id
                            """,
                            int(guild_id), channel_id
                        )

                return web.json_response({
                    "success": True,
                    "message": "Successfully updated automation settings"
                })

            except Exception as e:
                log.error(f"Error in specific automation update: {e}")
                return web.json_response(
                    {"error": f"Failed to update automation: {str(e)}"}, 
                    status=500
                )

        except Exception as e:
            log.error(f"Error updating automation: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/socials")
    @requires_auth
    async def socials(self: "Network", request: Request) -> Response:
        try:
            target = request.headers.get("X-USER-ID")
            if not target:
                return web.json_response({"error": "Missing X-USER-ID header"}, status=400)

            user = None
            
            if target.isdigit():
                try:
                    user = await self.bot.fetch_user(int(target))
                    user_data = await self.bot.http.get_user(user.id)
                    if 'banner' in user_data:
                        user._banner = user_data['banner']
                except discord.NotFound:
                    pass
            
            if not user:
                support_guild = self.bot.get_guild(892675627373699072)
                if support_guild:
                    members = [m for m in support_guild.members if m.name.lower() == target.lower()]
                    if members:
                        user = await self.bot.fetch_user(members[0].id)
                        user_data = await self.bot.http.get_user(user.id)
                        if 'banner' in user_data:
                            user._banner = user_data['banner']
                    else:
                        for guild in self.bot.guilds:
                            members = [m for m in guild.members if m.name.lower() == target.lower()]
                            if members:
                                user = await self.bot.fetch_user(members[0].id)
                                user_data = await self.bot.http.get_user(user.id)
                                if 'banner' in user_data:
                                    user._banner = user_data['banner']
                                break
            
            if not user:
                return web.json_response({"error": "User not found"}, status=404)

            socials, friends, links = await asyncio.gather(
                self.bot.db.fetchrow(
                    "SELECT * FROM socials WHERE user_id = $1",
                    user.id  
                ),
                self.bot.db.fetch(
                    "SELECT friends FROM socials_details WHERE user_id = $1",
                    user.id
                ),
                self.bot.db.fetch(
                    "SELECT * FROM social_links WHERE user_id = $1",
                    user.id
                )
            )

            friend_ids = [row['friends'] for row in friends]
            if friend_ids:
                try:
                    friend_users = await asyncio.gather(*[
                        self.bot.fetch_user(friend_id) 
                        for friend_id in friend_ids
                    ], return_exceptions=True)
                except:
                    friend_users = []
            else:
                friend_users = []

            friends_data = []
            for i, friend_user in enumerate(friend_users):
                if isinstance(friend_user, Exception):
                    friends_data.append({
                        "id": str(friend_ids[i]),
                        "name": "Unknown User",
                        "avatar": None
                    })
                else:
                    friends_data.append({
                        "id": str(friend_user.id),
                        "name": friend_user.name,
                        "avatar": str(friend_user.display_avatar.url) if friend_user.display_avatar else None
                    })

            def parse_emojis(text):
                if not text:
                    return None, text
                
                custom_match = re.search(r'<(a)?:([a-zA-Z0-9_]+):(\d+)>', text)
                if custom_match:
                    animated, name, emoji_id = custom_match.groups()
                    ext = 'gif' if animated else 'png'
                    return {
                        "name": name,
                        "id": emoji_id,
                        "url": f"https://cdn.discordapp.com/emojis/{emoji_id}.{ext}?size=48"
                    }, re.sub(r'<(a)?:([a-zA-Z0-9_]+):(\d+)>', name, text)
                
                unicode_match = re.search(r'[\U0001F300-\U0001F9FF]', text)
                if unicode_match:
                    emoji = unicode_match.group(0)
                    return {
                        "name": emoji,
                        "unicode": emoji,
                        "url": None
                    }, text
                
                return None, text

            bio_emoji, clean_bio = None, None
            
            if socials and socials.get('bio'):
                bio_emoji, clean_bio = parse_emojis(socials['bio'])

            badges = []
            staff_eligible = False
            presence_data = None

            support_guild = self.bot.get_guild(892675627373699072)
            if support_guild:
                support_member = support_guild.get_member(user.id)
                if support_member:
                    role_badges = {
                        1265473601755414528: ["developer", "owner"],
                        1264110559989862406: ["support"],
                        1323255508609663098: ["trial"],
                        1325007612797784144: ["mod"],
                        1318054098666389534: ["donor1"],
                        1320428924215496704: ["donor4"]
                    }

                    for role_id, badge_types in role_badges.items():
                        if any(role.id == role_id for role in support_member.roles):
                            badges.extend(badge_types)
                            if role_id not in [1318054098666389534, 1320428924215496704]:
                                staff_eligible = True

                    if staff_eligible:
                        badges.append("staff")

                    if support_member.status != discord.Status.offline:
                        activities_data = []
                        for activity in support_member.activities:
                            details_emoji, clean_details = parse_emojis(activity.details if hasattr(activity, 'details') else None)
                            state_emoji, clean_state = parse_emojis(activity.state if hasattr(activity, 'state') else None)
                            
                            activity_data = {
                                "name": activity.name,
                                "type": str(activity.type),
                                "details": clean_details,
                                "details_emoji": details_emoji,
                                "state": clean_state,
                                "state_emoji": state_emoji,
                                "emoji": str(activity.emoji) if hasattr(activity, 'emoji') and activity.emoji else None
                            }

                            if isinstance(activity, discord.Spotify):
                                activity_data.update({
                                    "details": activity.title,
                                    "state": f"{activity.artist} - {activity.album}",
                                    "album_cover_url": activity.album_cover_url,
                                    "track_url": f"https://open.spotify.com/track/{activity.track_id}",
                                    "duration": activity.duration.total_seconds(),
                                    "start": activity.start.timestamp() if activity.start else None,
                                    "end": activity.end.timestamp() if activity.end else None
                                })
                            elif isinstance(activity, discord.Activity):
                                if activity.application_id:
                                    activity_data.update({
                                        "application_id": str(activity.application_id),
                                        "large_image": activity.large_image_url,
                                        "small_image": activity.small_image_url,
                                        "large_text": activity.large_image_text,
                                        "small_text": activity.small_image_text
                                    })

                            activities_data.append(activity_data)

                        presence_data = {
                            "status": str(support_member.status),
                            "activities": activities_data
                        }

            profile_gradient_colors = await self.bot.db.fetch(
                """
                SELECT color, position 
                FROM socials_gradients 
                WHERE user_id = $1 
                ORDER BY position
                """,
                user.id
            )

            element_types = ["text_underline", "bold_text", "status", "bio", "social_icons"]
            element_colors = {}
            
            for element in element_types:
                if socials:
                    color_type = socials.get(f'{element}_color_type', 'linear')
                    if color_type == 'linear':
                        element_colors[element] = {
                            "type": "linear",
                            "color": socials.get(f'{element}_linear_color', '#ffffff')
                        }
                    else:
                        gradient_name = socials.get(f'{element}_gradient_name')
                        if gradient_name:
                            gradient_colors = await self.bot.db.fetch(
                                """
                                SELECT color, position 
                                FROM socials_saved_gradients 
                                WHERE user_id = $1 AND name = $2 
                                ORDER BY position
                                """,
                                user.id, gradient_name
                            )
                            element_colors[element] = {
                                "type": "gradient",
                                "name": gradient_name,
                                "colors": [
                                    {
                                        "color": color['color'],
                                        "position": color['position']
                                    } for color in gradient_colors
                                ]
                            }
                        else:
                            element_colors[element] = {
                                "type": "linear",
                                "color": "#ffffff"
                            }
                else:
                    element_colors[element] = {
                        "type": "linear",
                        "color": "#ffffff"
                    }

            response_data = {
                "user": {
                    "id": str(user.id),
                    "name": user.name,
                    "avatar": str(user.display_avatar.url) if user.display_avatar else str(user.default_avatar.url),
                    "banner": user.banner.url if user.banner else None,
                    "created_at": user.created_at.isoformat(),
                    "avatar_decoration_data": {
                        "sku_id": user.avatar_decoration_sku_id if hasattr(user, 'avatar_decoration_sku_id') else None,
                        "asset": user.avatar_decoration.key if hasattr(user, 'avatar_decoration') and user.avatar_decoration else None,
                        "expires_at": None
                    } if hasattr(user, 'avatar_decoration') and user.avatar_decoration else None,
                    "global_name": user.global_name if hasattr(user, 'global_name') else None,
                    "display_name": user.display_name if hasattr(user, 'display_name') else user.name,
                    "public_flags": user.public_flags.value if hasattr(user, 'public_flags') else None,
                    "status_indicator": {
                        "offset": True if hasattr(user, 'avatar_decoration') and user.avatar_decoration else False,
                        "position": "bottom-right"
                    }
                },
                "profile_image": socials.get('profile_image') if socials else None,
                "presence": presence_data if presence_data and (not socials or socials.get('show_activity', True)) else None,
                "badges": badges or [],
                "bio": clean_bio,
                "bio_emoji": bio_emoji,
                "friends": friends_data if not socials or socials.get('show_friends', True) else None,
                "show_friends": socials.get('show_friends', True) if socials else True,
                "show_activity": socials.get('show_activity', True) if socials else True,
                "background_url": socials.get('background_url') if socials else None,
                "colors": {
                    "profile": {
                        "type": socials.get('profile_color', 'linear') if socials else 'linear',
                        "linear_color": socials.get('linear_color', '#ffffff') if socials else '#ffffff',
                        "gradient_colors": [
                            {
                                "color": color['color'],
                                "position": color['position']
                            } for color in profile_gradient_colors  
                        ] if profile_gradient_colors else []
                    },
                    "elements": element_colors
                },
                "glass_effect": socials.get('glass_effect', False) if socials else False,
                "discord_guild": socials.get('discord_guild') if socials else None,
                "click": {
                    "enabled": socials.get('click_enabled', False) if socials else False,
                    "text": socials.get('click_text', 'Click to enter...') if socials else 'Click to enter...'
                },
                "audio": {
                    "url": socials.get('audio_url') if socials else None,
                    "title": socials.get('audio_title') if socials else None
                },
                "links": [
                    {
                        "type": link['type'],
                        "url": link['url']
                    } for link in links
                ] if links else []
            }

            return web.json_response(response_data)

        except Exception as e:
            log.error(f"Error fetching social data: {e}", exc_info=True)  
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/report", ["POST"])
    async def report_user(self: "Network", request: Request) -> Response:
        """Handle user reports for social profiles."""
        try:
            data = await request.json()
            
            required_fields = {
                "username_reported": str,
                "reason": str,
                "description": str,
                "reporter_email": str
            }
            
            for field, field_type in required_fields.items():
                if field not in data:
                    return web.json_response(
                        {"error": f"Missing required field: {field}"}, 
                        status=400
                    )
                if not isinstance(data[field], field_type):
                    return web.json_response(
                        {"error": f"Field {field} must be a {field_type.__name__}"}, 
                        status=400
                    )

            auth_header = request.headers.get("Authorization")
            token = auth_header.split(" ")[1]
            reporter_data = await self.bot.db.fetchrow(
                """
                SELECT user_id
                FROM access_tokens 
                WHERE token = $1 
                AND expires_at > CURRENT_TIMESTAMP
                """,
                token
            )

            existing_report = await self.bot.db.fetchrow(
                """
                SELECT id FROM reports
                WHERE reporter_id = $1 
                AND username_reported = $2
                AND reviewed = false
                """,
                reporter_data['user_id'],
                data['username_reported']
            )
            
            if existing_report:
                return web.json_response(
                    {"error": "You already have an active report for this user"}, 
                    status=400
                )

            try:
                reporter = await self.bot.fetch_user(reporter_data['user_id'])
            except discord.NotFound:
                return web.json_response(
                    {"error": "Reporter Discord account not found"}, 
                    status=404
                )

            report_id = await self.bot.db.fetchval(
                """
                INSERT INTO reports (
                    reporter_id,
                    reporter_name,
                    reporter_email,
                    username_reported,
                    reason,
                    description,
                    created_at,
                    reviewed
                )
                VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP, false)
                RETURNING id
                """,
                reporter.id,
                str(reporter),
                data['reporter_email'],
                data['username_reported'],
                data['reason'],
                data['description']
            )

            channel = self.bot.get_channel(1325762712332009473) 
            if channel:
                embed = discord.Embed(
                    title="New User Report",
                    color=discord.Color.red(),
                    timestamp=discord.utils.utcnow()
                )
                embed.add_field(name="Report ID", value=str(report_id), inline=True)
                embed.add_field(name="Reporter", value=f"{reporter.mention} ({reporter.id})", inline=True)
                embed.add_field(name="Email", value=data['reporter_email'], inline=True)
                embed.add_field(name="Reported User", value=data['username_reported'], inline=True)
                embed.add_field(name="Reason", value=data['reason'], inline=False)
                embed.add_field(name="Description", value=data['description'], inline=False)
                await channel.send(embed=embed)

            return web.json_response({
                "success": True,
                "report_id": report_id
            })

        except Exception as e:
            log.error(f"Error processing report: {e}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/audio")
    async def audio_data(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                "SELECT user_id FROM access_tokens WHERE token = $1 AND expires_at > CURRENT_TIMESTAMP",
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            recent_tracks = await self.bot.db.fetch(
                "SELECT * FROM audio.recently_played WHERE guild_id = $1 AND user_id = $2 ORDER BY played_at DESC LIMIT 50",
                int(guild_id),
                user_data['user_id']
            )

            playlists = await self.bot.db.fetch(
                "SELECT * FROM audio.playlists WHERE guild_id = $1 AND user_id = $2 ORDER BY added_at DESC",
                int(guild_id),
                user_data['user_id']
            )

            playlist_tracks = await self.bot.db.fetch(
                "SELECT * FROM audio.playlist_tracks WHERE guild_id = $1 AND user_id = $2 ORDER BY added_at DESC",
                int(guild_id),
                user_data['user_id']
            )

            playlists_data = []
            for playlist in playlists:
                tracks = [
                    {
                        "title": track['track_title'],
                        "uri": track['track_uri'],
                        "author": track['track_author'],
                        "album": track['album_name'],
                        "artwork_url": track['artwork_url'],
                        "added_at": track['added_at'].isoformat()
                    }
                    for track in playlist_tracks
                    if track['playlist_url'] == playlist['playlist_url']
                ]
                
                playlists_data.append({
                    "name": playlist['playlist_name'],
                    "url": playlist['playlist_url'],
                    "track_count": playlist['track_count'],
                    "added_at": playlist['added_at'].isoformat(),
                    "tracks": tracks
                })

            return web.json_response({
                "recently_played": [
                    {
                        "title": track['track_title'],
                        "uri": track['track_uri'],
                        "author": track['track_author'],
                        "artwork_url": track['artwork_url'],
                        "playlist": {
                            "name": track['playlist_name'],
                            "url": track['playlist_url']
                        } if track['playlist_name'] else None,
                        "played_at": track['played_at'].isoformat()
                    }
                    for track in recent_tracks
                ],
                "playlists": playlists_data
            })

        except Exception as e:
            log.error(f"Error fetching audio data: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/voice")
    @ratelimit(5, 60)
    async def voice_info(self: "Network", request: Request) -> Response:
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                "SELECT user_id FROM access_tokens WHERE token = $1 AND expires_at > CURRENT_TIMESTAMP",
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild_id = request.headers.get("X-GUILD-ID")
            if not guild_id:
                return web.json_response(
                    {"error": "Missing X-GUILD-ID header"}, 
                    status=400
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            member = guild.get_member(user_data['user_id'])
            if not member:
                return web.json_response(
                    {"error": "Member not found in guild"}, 
                    status=404
                )

            if not member.voice:
                return web.json_response(
                    {"error": "You're not in a voice channel"}, 
                    status=400
                )

            voice_client = guild.voice_client
            current_channel = member.voice.channel

            everyone_role = guild.default_role
            is_private = not current_channel.permissions_for(everyone_role).view_channel

            queue_mode = "none"
            if voice_client and hasattr(voice_client, 'queue'):
                if voice_client.queue.loop_mode == LoopMode.TRACK:
                    queue_mode = "track"
                elif voice_client.queue.loop_mode == LoopMode.QUEUE:
                    queue_mode = "queue"
                if voice_client.queue.shuffle:
                    queue_mode = "shuffle"

            available_channels = [
                {
                    "id": vc.id,
                    "name": vc.name,
                    "user_limit": vc.user_limit,
                    "member_count": len(vc.members),
                    "is_private": not vc.permissions_for(guild.default_role).view_channel
                }
                for vc in guild.voice_channels
                if vc.permissions_for(member).connect
            ]

            return web.json_response({
                "current_channel": {
                    "id": current_channel.id,
                    "name": current_channel.name,
                    "connected": voice_client and voice_client.channel.id == current_channel.id if voice_client else False,
                    "is_private": is_private,
                    "listeners": [
                        {
                            "id": m.id,
                            "name": m.display_name,
                            "avatar": str(m.display_avatar.url),
                            "bot": m.bot,
                            "speaking": False,
                            "can_manage": True 
                        }
                        for m in current_channel.members
                    ],
                    "queue_mode": queue_mode
                },
                "available_channels": available_channels
            })

        except Exception as e:
            log.error(f"Error fetching voice info: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    @route("/topgg/webhook", ["POST"])
    async def topgg_webhook(self: "Network", request: Request) -> Response:
        """Handle incoming votes from top.gg"""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or auth_header != "":
                return web.json_response(
                    {"error": "Invalid authorization"}, 
                    status=401
                )

            data = await request.json()
            if not data or "user" not in data:
                return web.json_response(
                    {"error": "Invalid payload"}, 
                    status=400
                )

            user_id = data["user"]
            await self.bot.db.execute(
                """
                INSERT INTO user_votes (user_id, last_vote_time)
                VALUES ($1, NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET last_vote_time = NOW()
                """,
                int(user_id)
            )

            try:
                user = await self.bot.fetch_user(int(user_id))
                if user:
                    channel = self.bot.get_channel(1325865864834187334)
                    if channel:
                        embed = discord.Embed(
                            title="New Vote",
                            description=(
                                f"🎉 {user.mention} (`{user.id}`) has voted for the bot!\n\n"
                                "Access to donator-only commands granted for 6 hours.\n"
                                "Vote again in 12 hours to maintain access."
                            ),
                            color=0x2f3136,
                            timestamp=datetime.now()
                        )
                        embed.set_footer(text="Tip: Donators get permanent access to these commands!")
                        await channel.send(embed=embed)
            except Exception as e:
                log.error(f"Error sending vote notification: {e}")

            return web.json_response({"success": True})

        except Exception as e:
            log.error(f"Error processing top.gg webhook: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/domains/verify", method="GET")
    @ratelimit(5, 60)
    @requires_auth
    async def verify_domain(self, request: Request) -> Response:
        """Verify domain DNS records"""
        domain = request.query.get('domain')
        if not domain:
            return web.json_response({"error": "Missing domain parameter"}, status=400)
        
        try:
            cname_records = await self.bot.loop.run_in_executor(
                None, 
                lambda: dns.resolver.resolve(domain, 'CNAME')
            )
            cname_valid = any(str(record.target).rstrip('.') == 'cname.evict.bot' for record in cname_records)
            
            if not cname_valid:
                return web.json_response({
                    "error": "CNAME record not properly configured",
                    "details": "Domain should point to cname.evict.bot"
                }, status=400)
                
            txt_records = await self.bot.loop.run_in_executor(
                None, 
                lambda: dns.resolver.resolve(f'_evict-verify.{domain}', 'TXT')
            )
            
            owner = None
            for record in txt_records:
                txt_value = str(record).strip('"')
                if txt_value.startswith("evict-verify="):
                    owner = txt_value.split("=")[1]
                    break
                    
            if not owner:
                return web.json_response({
                    "error": "TXT record not properly configured",
                    "details": "Missing or invalid verification record"
                }, status=400)
                
            return web.json_response({
                "success": True,
                "domain": domain,
                "owner": owner
            })
            
        except dns.resolver.NXDOMAIN:
            return web.json_response({
                "error": "Domain not found",
                "details": "Could not resolve DNS records"
            }, status=404)
        except Exception as e:
            return web.json_response({
                "error": "Verification failed",
                "details": str(e)
            }, status=500)

    @route("/verification/status/{guild_id}")
    @ratelimit(5, 60)
    async def verification_status(self: "Network", request: Request) -> Response:
        """Get verification settings and status for a guild."""
        try:
            guild_id = request.match_info["guild_id"]
            
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                "SELECT user_id FROM access_tokens WHERE token = $1 AND expires_at > CURRENT_TIMESTAMP",
                token
            )
            
            if not user_data:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            settings = await self.bot.db.fetchrow(
                """
                SELECT level, kick_after, ratelimit, antialt, bypass_until, prevent_vpn
                FROM guild_verification 
                WHERE guild_id = $1
                """,
                int(guild_id)
            )
            
            if not settings:
                return web.json_response(
                    {"error": "Verification not configured for this guild"}, 
                    status=404
                )

            verification_methods = {
                1: {
                    "type": "email",
                    "name": "Email Verification",
                    "description": "Verify your account using your email address"
                },
                2: {
                    "type": "oauth",
                    "name": "LastFM Verification",
                    "description": "Verify using your LastFM account",
                    "provider": "lastfm"
                },
                3: {
                    "type": "captcha",
                    "name": "CAPTCHA Verification",
                    "description": "Complete a CAPTCHA challenge"
                },
                4: {
                    "type": "questions",
                    "name": "Custom Questions",
                    "description": "Answer server-specific questions"
                }
            }

            return web.json_response({
                "guild_id": guild_id,
                "guild_name": guild.name,
                "guild_icon": str(guild.icon.url) if guild.icon else None,
                "verification": {
                    "method": verification_methods[settings["level"]],
                    "settings": {
                        "auto_kick": settings["kick_after"],
                        "rate_limit": settings["ratelimit"],
                        "anti_alt": settings["antialt"],
                        "bypass_until": settings["bypass_until"].isoformat() if settings["bypass_until"] else None,
                        "block_vpn": settings["prevent_vpn"] if settings["prevent_vpn"] is not None else False
                    }
                }
            })

        except Exception as e:
            log.error(f"Error fetching verification status: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/verification/user/{guild_id}/{user_id}")
    @ratelimit(5, 60)
    async def verification_user_status(self: "Network", request: Request) -> Response:
        """Get verification status and requirements for a user."""
        try:
            guild_id = request.match_info["guild_id"]
            target_user_id = request.match_info["user_id"]
            
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                "SELECT user_id FROM access_tokens WHERE token = $1 AND expires_at > CURRENT_TIMESTAMP",
                token
            )
            
            if not user_data or str(user_data["user_id"]) != target_user_id:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response({
                    "code": "GUILD_NOT_FOUND",
                    "message": "Guild not found"
                }, status=404)

            member = guild.get_member(int(target_user_id))
            if not member:
                return web.json_response({
                    "code": "NOT_IN_GUILD",
                    "message": "User is not in the guild"
                })

            settings = await self.bot.db.fetchrow(
                """
                SELECT level, kick_after, ratelimit, antialt, bypass_until
                FROM guild_verification 
                WHERE guild_id = $1
                """,
                int(guild_id)
            )
            
            if not settings:
                return web.json_response({
                    "code": "NO_VERIFICATION",
                    "message": "Verification is not configured for this guild"
                })

            bypass_roles = await self.bot.db.fetch(
                "SELECT role_id FROM verification_bypass_roles WHERE guild_id = $1",
                int(guild_id)
            )
            
            for role_data in bypass_roles:
                role = guild.get_role(role_data["role_id"])
                if role and role in member.roles:
                    return web.json_response({
                        "code": "BYPASS_ROLE",
                        "message": "User has a bypass role",
                        "role": {
                            "id": str(role.id),
                            "name": role.name
                        }
                    })

            if settings["bypass_until"] and settings["bypass_until"] > datetime.now(timezone.utc):
                return web.json_response({
                    "code": "BYPASS_TEMPORARY",
                    "message": "Server has temporary verification bypass",
                    "expires_at": settings["bypass_until"].isoformat()
                })

            if settings["ratelimit"]:
                recent_attempts = await self.bot.db.fetchval(
                    """
                    SELECT COUNT(*) FROM verification_attempts 
                    WHERE user_id = $1 AND guild_id = $2 
                    AND attempt_time > NOW() - INTERVAL '1 hour'
                    """,
                    int(target_user_id),
                    int(guild_id)
                )
                
                if recent_attempts >= settings["ratelimit"]:
                    return web.json_response({
                        "code": "RATE_LIMITED",
                        "message": "Too many verification attempts",
                        "retry_after": 3600 
                    })

            if settings["antialt"]:
                account_age = datetime.now(timezone.utc) - member.created_at
                if account_age.days < 7: 
                    return web.json_response({
                        "code": "ACCOUNT_TOO_NEW",
                        "message": "Account is too new",
                        "account_age": account_age.days
                    })

            return web.json_response({
                "code": "VERIFICATION_REQUIRED",
                "message": "User needs to complete verification",
                "user": {
                    "id": str(member.id),
                    "name": member.name,
                    "avatar": str(member.display_avatar.url)
                },
                "method": {
                    "type": ["email", "oauth", "captcha", "questions"][settings["level"] - 1],
                    "timeout": 600  
                }
            })

        except Exception as e:
            log.error(f"Error checking user verification status: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/verification/start/{guild_id}/{user_id}", method="POST")
    @ratelimit(5, 60)
    async def start_verification(self: "Network", request: Request) -> Response:
        """Initialize verification process."""
        try:
            guild_id = request.match_info["guild_id"]
            target_user_id = request.match_info["user_id"]
            
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                "SELECT user_id FROM access_tokens WHERE token = $1 AND expires_at > CURRENT_TIMESTAMP",
                token
            )
            
            if not user_data or str(user_data["user_id"]) != target_user_id:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            settings = await self.bot.db.fetchrow(
                """
                SELECT level, verified_role_id FROM guild_verification 
                WHERE guild_id = $1
                """,
                int(guild_id)
            )
            
            if not settings:
                return web.json_response(
                    {"error": "Verification not configured for this guild"}, 
                    status=404
                )

            if not settings['verified_role_id']:
                return web.json_response(
                    {"error": "No verification role set for this guild"}, 
                    status=404
                )

            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                return web.json_response(
                    {"error": "Guild not found"}, 
                    status=404
                )

            role = guild.get_role(settings['verified_role_id'])
            if not role:
                return web.json_response(
                    {"error": "Verification role no longer exists"}, 
                    status=404
                )

            member = guild.get_member(int(target_user_id))
            if not member:
                return web.json_response(
                    {"error": "User not in guild"}, 
                    status=404
                )

            if role in member.roles:
                return web.json_response({
                    "error": "User is already verified",
                    "code": "ALREADY_VERIFIED"
                }, status=400)

            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
            
            await self.bot.db.execute(
                """
                INSERT INTO verification_sessions (
                    session_token,
                    user_id,
                    guild_id,
                    method,
                    expires_at,
                    created_at
                ) VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
                """,
                session_token,
                int(target_user_id),
                int(guild_id),
                settings['level'],
                expires_at
            )

            if settings['level'] == 1:
                code = ''.join(random.choices(string.digits, k=6))
                await self.bot.db.execute(
                    """
                    INSERT INTO verification_email_codes (
                        session_token,
                        code,
                        expires_at
                    ) VALUES ($1, $2, $3)
                    """,
                    session_token,
                    code,
                    expires_at
                )
                
                return web.json_response({
                    "session": session_token,
                    "expires_at": expires_at.isoformat(),
                    "verification_url": f"https://evict.bot/verify/email/{session_token}"
                })

            elif settings['level'] == 2:
                return web.json_response({
                    "session": session_token,
                    "expires_at": expires_at.isoformat(),
                    "oauth_url": f"https://www.last.fm/api/auth?api_key={self.bot.config.lastfm.key}&cb=https://evict.bot/verify/oauth/callback&state={session_token}"
                })

            elif settings['level'] == 3: 
                return web.json_response({
                    "session": session_token,
                    "expires_at": expires_at.isoformat(),
                    "captcha_url": f"https://evict.bot/verify/captcha/{session_token}"
                })

            elif settings['level'] == 4:  
                questions = await self.bot.db.fetch(
                    """
                    SELECT id, question, options, is_text
                    FROM verification_questions
                    WHERE guild_id = $1
                    ORDER BY RANDOM()
                    """,
                    int(guild_id)
                )
                
                if not questions:
                    return web.json_response({
                        "error": "No verification questions configured",
                        "code": "NO_QUESTIONS"
                    }, status=404)
                
                has_text_questions = any(q['is_text'] for q in questions)
                
                await self.bot.db.execute(
                    """
                    INSERT INTO verification_question_sessions (
                        session_token,
                        question_ids,
                        requires_review,
                        expires_at
                    ) VALUES ($1, $2, $3, $4)
                    """,
                    session_token,
                    [q['id'] for q in questions],
                    has_text_questions,
                    expires_at
                )
                
                return web.json_response({
                    "session": session_token,
                    "expires_at": expires_at.isoformat(),
                    "requires_review": has_text_questions,
                    "questions": [
                        {
                            "id": q['id'],
                            "question": q['question'],
                            "type": "text" if q['is_text'] else "choice",
                            "options": q['options'] if not q['is_text'] else None,
                            "max_length": 1000 if q['is_text'] else None
                        } for q in questions
                    ]
                })

        except Exception as e:
            log.error(f"Error starting verification: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/verification/verify/{guild_id}/{user_id}", method="POST")
    @ratelimit(5, 60)
    async def verify_user(self: "Network", request: Request) -> Response:
        """Handle verification submission."""
        try:
            guild_id = request.match_info["guild_id"]
            target_user_id = request.match_info["user_id"]
            
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing authorization"}, 
                    status=401
                )
            
            token = auth_header.split(" ")[1]
            user_data = await self.bot.db.fetchrow(
                "SELECT user_id FROM access_tokens WHERE token = $1 AND expires_at > CURRENT_TIMESTAMP",
                token
            )
            
            if not user_data or str(user_data["user_id"]) != target_user_id:
                return web.json_response(
                    {"error": "Invalid or expired token"}, 
                    status=401
                )

            data = await request.json()
            session_token = data.get('session')
            
            if not session_token:
                return web.json_response(
                    {"error": "Missing session token"}, 
                    status=400
                )

            session = await self.bot.db.fetchrow(
                """
                SELECT method, expires_at
                FROM verification_sessions
                WHERE session_token = $1 
                AND user_id = $2 
                AND guild_id = $3
                """,
                session_token,
                int(target_user_id),
                int(guild_id)
            )
            
            if not session:
                return web.json_response(
                    {"error": "Invalid session"}, 
                    status=400
                )
                
            if session['expires_at'] < datetime.now(timezone.utc):
                return web.json_response(
                    {"error": "Session expired"}, 
                    status=400
                )

            verification_success = False
            
            if session['method'] == 1:
                
                code = data.get('code')
                session_token = data.get('session')         
                
                if not code or not session_token:
                    log.warning(f"Missing code or session for verification")
                    return web.json_response({
                        "success": False,
                        "message": "Verification failed",
                        "error": "Missing required fields",
                        "code": "MISSING_FIELDS"
                    }, status=400)

                stored_code = await self.bot.db.fetchval(
                    """
                    SELECT code FROM verification_email_codes
                    WHERE session_token = $1
                    """,
                    session_token
                )
                
                
                if not stored_code:
                    log.warning(f"Code not found for session {session_token}")
                    return web.json_response({
                        "success": False,
                        "message": "Verification failed",
                        "error": "Code expired or not found",
                        "code": "CODE_NOT_FOUND"
                    }, status=400)

                if code != stored_code:
                    log.warning(f"Invalid verification code for session {session_token}")
                    return web.json_response({
                        "success": False,
                        "message": "Verification failed",
                        "error": "Invalid verification code",
                        "code": "INVALID_CODE"
                    }, status=400)

                verification_success = True

            elif session['method'] == 2:
                lastfm_token = data.get('oauth_token')
                if not lastfm_token:
                    return web.json_response(
                        {"error": "Missing OAuth token"}, 
                        status=400
                    )
                    
                async with self.bot.session.get(
                    "http://ws.audioscrobbler.com/2.0/",
                    params={
                        "method": "auth.getSession",
                        "api_key": self.bot.config.lastfm.key,
                        "token": lastfm_token,
                        "format": "json"
                    }
                ) as resp:
                    if resp.status == 200:
                        verification_success = True

            elif session['method'] == 3: 
                raw_data = await request.text()
                
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError as e:
                    log.error(f"Failed to parse CAPTCHA JSON data: {e}")
                    return web.json_response({
                        "success": False,
                        "message": "Verification failed",
                        "error": "Invalid JSON data",
                        "code": "INVALID_JSON"
                    }, status=400)
                
                captcha_token = data.get('token')
                captcha_success = data.get('success')
                challenge_ts = data.get('challenge_ts')
                
                if not captcha_token or not captcha_success or not challenge_ts:
                    log.warning("Missing required CAPTCHA fields")
                    return web.json_response({
                        "success": False,
                        "message": "Verification failed",
                        "error": "Missing required CAPTCHA fields",
                        "code": "MISSING_CAPTCHA_FIELDS"
                    }, status=400)
                
                if not captcha_success:
                    log.warning("CAPTCHA verification reported as unsuccessful")
                    return web.json_response({
                        "success": False,
                        "message": "Verification failed",
                        "error": "CAPTCHA verification unsuccessful",
                        "code": "CAPTCHA_FAILED"
                    }, status=400)
                    
                if not re.match(r'^[0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[0-9]{12}$', captcha_token):
                    log.warning(f"Invalid CAPTCHA token format: {captcha_token}")
                    return web.json_response({
                        "success": False,
                        "message": "Verification failed",
                        "error": "Invalid CAPTCHA token format",
                        "code": "INVALID_CAPTCHA"
                    }, status=400)
                
                if not captcha_token.startswith('10000000'):
                    log.warning(f"Failed CAPTCHA verification with token: {captcha_token}")
                    return web.json_response({
                        "success": False,
                        "message": "Verification failed",
                        "error": "CAPTCHA verification failed",
                        "code": "CAPTCHA_FAILED"
                    }, status=400)
                
                verification_success = True

            elif session['method'] == 4: 
                answers = data.get('answers')
                if not answers:
                    return web.json_response(
                        {"error": "Missing answers"}, 
                        status=400
                    )

                session_data = await self.bot.db.fetchrow(
                    """
                    SELECT question_ids, requires_review
                    FROM verification_question_sessions
                    WHERE session_token = $1
                    """,
                    session_token
                )

                if session_data['requires_review']:
                    await self.bot.db.execute(
                        """
                        INSERT INTO verification_pending_reviews (
                            session_token,
                            user_id,
                            guild_id,
                            answers,
                            submitted_at
                        ) VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                        """,
                        session_token,
                        int(target_user_id),
                        int(guild_id),
                        answers
                    )

                    try:
                        log_channel = await self.bot.db.fetchval(
                            "SELECT log_channel_id FROM guild_verification WHERE guild_id = $1",
                            int(guild_id)
                        )
                        channel = self.bot.get_channel(log_channel)
                        if channel:
                            embed = discord.Embed(
                                title="Verification Review Required",
                                description=f"User: <@{target_user_id}>",
                                color=0x2f3136
                            )
                            for qid, answer in answers.items():
                                embed.add_field(
                                    name=f"Question {qid}",
                                    value=f"Answer: {answer}",
                                    inline=False
                                )
                            await channel.send(embed=embed)
                    except Exception as e:
                        log.error(f"Failed to send review notification: {e}", exc_info=True)

                    return web.json_response({
                        "success": True,
                        "message": "Answers submitted for review",
                        "status": "pending_review"
                    })

                questions = await self.bot.db.fetch(
                    """
                    SELECT id, correct_answer, is_text
                    FROM verification_questions
                    WHERE id = ANY($1)
                    """,
                    session_data['question_ids']
                )

                verification_success = all(
                    str(answer) == str(q['correct_answer'])
                    for answer, q in zip(answers, questions)
                    if not q['is_text']
                )

            await self.bot.db.execute(
                """
                INSERT INTO verification_attempts (
                    user_id,
                    guild_id,
                    success,
                    method
                ) VALUES ($1, $2, $3, $4)
                """,
                int(target_user_id),
                int(guild_id),
                verification_success,
                session['method']
            )

            if not verification_success:
                error_reason = "Unknown error"
                
                if session['method'] == 1:
                    if not code:
                        error_reason = "Code expired or not found"
                    else:
                        error_reason = "Invalid verification code"
                elif session['method'] == 2:  
                    error_reason = "OAuth verification failed"
                elif session['method'] == 3:  
                    error_reason = "Invalid CAPTCHA response"
                elif session['method'] == 4:  
                    error_reason = "Incorrect answers provided"
                
                return web.json_response({
                    "success": False,
                    "message": "Verification failed",
                    "error": error_reason,
                    "code": f"VERIFICATION_FAILED_{session['method']}"
                }, status=400)

            guild = self.bot.get_guild(int(guild_id))
            member = guild.get_member(int(target_user_id))
            
            if not member:
                return web.json_response({
                    "error": "User no longer in guild"
                }, status=400)

            try:
                role_id = await self.bot.db.fetchval(
                    "SELECT verified_role_id FROM guild_verification WHERE guild_id = $1",
                    int(guild_id)
                )

                if role_id:
                    role = guild.get_role(role_id)
                    if role:
                        
                        if role in member.roles:
                            log.warning(f"Member already has verification role")
                            return web.json_response({
                                "success": False,
                                "message": "Verification failed",
                                "error": "User is already verified",
                                "code": "ALREADY_VERIFIED"
                            }, status=400)
                            
                        await member.add_roles(role, reason="Verification successful")
                    else:
                        log.warning(f"Verification role {role_id} not found in guild {guild_id}")
                else:
                    log.warning(f"No verification role configured for guild {guild_id}")
                
                await self.bot.db.execute(
                    "DELETE FROM verification_sessions WHERE session_token = $1",
                    session_token
                )
                
                return web.json_response({
                    "success": True,
                    "message": "Verification successful"
                })

            except discord.Forbidden as e:
                log.error(f"Permission error adding role: {e}")
                return web.json_response({
                    "success": False,
                    "message": "Verification failed",
                    "error": "Bot missing permissions to assign role",
                    "code": "MISSING_PERMISSIONS"
                }, status=403)
            except Exception as e:
                log.error(f"Error adding role: {e}")
                return web.json_response({
                    "success": False,
                    "message": "Verification failed",
                    "error": "Failed to assign verification role",
                    "code": "ROLE_ASSIGNMENT_FAILED"
                }, status=500)

        except Exception as e:
            log.error(f"Error processing verification: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/verification/email/code", method="POST")
    @ratelimit(5, 60)
    async def store_email_code(self: "Network", request: Request) -> Response:
        """Store email verification code sent from frontend."""
        try:
            data = await request.json()
            session_token = data.get('session')
            code = data.get('code')
            
            if not session_token or not code:
                return web.json_response(
                    {"error": "Missing required fields"}, 
                    status=400
                )

            await self.bot.db.execute(
                """
                INSERT INTO verification_email_codes (
                    session_token,
                    code,
                    expires_at
                ) VALUES ($1, $2, NOW() + INTERVAL '10 minutes')
                ON CONFLICT (session_token) DO UPDATE 
                SET code = $2,
                    expires_at = NOW() + INTERVAL '10 minutes'
                """,
                session_token,
                code
            )
            
            return web.json_response({"success": True})

        except Exception as e:
            log.error(f"Error storing email code: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/verification/email/code", method="GET")
    @ratelimit(5, 60)
    async def get_email_code(self: "Network", request: Request) -> Response:
        """Check if email verification code exists and is valid."""
        try:
            session = request.query.get('session')
            if not session:
                return web.json_response(
                    {"error": "Missing session parameter"}, 
                    status=400
                )

            code_data = await self.bot.db.fetchrow(
                """
                SELECT EXISTS (
                    SELECT 1 FROM verification_email_codes
                    WHERE session_token = $1 
                    AND expires_at > CURRENT_TIMESTAMP
                ) as exists
                """,
                session
            )
            
            return web.json_response({
                "exists": code_data['exists'],
                "valid": code_data['exists']
            })

        except Exception as e:
            log.error(f"Error checking email code: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    @route("/github/webhook", ["POST"])
    async def github_webhook(self, request: Request) -> Response:
        """Handle GitHub webhook events"""
        try:
            event_type = request.headers.get('X-GitHub-Event')
            if event_type == 'ping':
                return web.json_response({"message": "Pong!"})

            if event_type != 'push':
                return web.json_response(
                    {"error": "Only push events are handled"}, 
                    status=400
                )

            payload = await request.read()
        
            signature = request.headers.get('X-Hub-Signature-256')
            if not signature:
                return web.json_response(
                    {"error": "Missing signature"}, 
                    status=401
                )

            secret = "".encode()
            expected_signature = 'sha256=' + hmac.new(
                secret,
                payload,
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                log.warning(f"Invalid GitHub signature from {request.remote}")
                return web.json_response(
                    {"error": "Invalid signature"}, 
                    status=401
                )

            cache_key = f"github_webhook:{request.remote}"
            if await self.bot.redis.exists(cache_key):
                return web.json_response(
                    {"error": "Rate limited"}, 
                    status=429
                )
            await self.bot.redis.set(cache_key, "1", ex=2)  

            try:
                data = orjson.loads(payload)
            except Exception as e:
                log.error(f"Failed to parse webhook payload: {e}")
                return web.json_response(
                    {"error": "Invalid payload"}, 
                    status=400
                )

            repo_name = data.get('repository', {}).get('full_name')
            allowed_repos = CONFIG.get('github_allowed_repos', [])
            if repo_name not in allowed_repos:
                log.warning(f"Webhook received for unauthorized repo: {repo_name}")
                return web.json_response(
                    {"error": "Repository not authorized"}, 
                    status=403
                )

            try:
                event = GithubPushEvent.parse_obj(data)
                await event.send_message()
            except Exception as e:
                log.error(f"Error processing webhook: {e}", exc_info=True)
                return web.json_response(
                    {"error": "Failed to process webhook"}, 
                    status=500
                )

            return web.json_response({"success": True})

        except Exception as e:
            log.error(f"Webhook error: {e}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )
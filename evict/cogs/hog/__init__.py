from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import Evict


async def setup(bot: "Evict") -> None:
    from .hog import Hog

    # await bot.add_cog(Hog(bot))

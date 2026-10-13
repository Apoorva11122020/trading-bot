"""Real-time paper trading for NIFTY S2."""
from .journal import PAPER_STATE_KEY, PaperTradingJournal
from src.safety.live_gate import assert_paper_trading_only

from .nifty_s2_runner import NiftyS2PaperRunner, print_startup_banner
from .session import PaperTradingSession, PaperTradingStats

__all__ = [
    "PAPER_STATE_KEY",
    "PaperTradingJournal",
    "NiftyS2PaperRunner",
    "PaperTradingSession",
    "PaperTradingStats",
    "assert_paper_trading_only",
    "print_startup_banner",
]

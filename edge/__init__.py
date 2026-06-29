"""edge — disciplined intraday edge-discovery & validation pipeline.

Prime directive: find and HONESTLY MEASURE real expectancy; REJECT overfit or
cost-fragile strategies. Never present an edge that fails the hard gates as viable.

Canonical bar schema used throughout the package:
    A pandas DataFrame with a tz-aware DatetimeIndex named 'ts' in the EXCHANGE
    timezone (America/Chicago), sorted ascending, columns:
        open, high, low, close : float  (price)
        volume                 : float  (contracts or tick-count)
    Optional engineered columns: session ('RTH'|'ETH'), and feature columns.
"""

__version__ = "0.1.0"

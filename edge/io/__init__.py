"""IO layer: data ingestion from multiple backends into the canonical schema.

Backends, in order of reliability:
  1. nt_csv        — NinjaTrader 'Historical Data Export' CSVs   (RELIABLE; primary)
  2. ncd_reader    — NinjaTrader .ncd binary:
                        * DAY files  -> fully decoded (uncompressed, verified)
                        * MINUTE/TICK files -> proprietary compressed stream
                          (EXPERIMENTAL; must be validated bit-exact vs a CSV
                          export before its output is trusted in research)
  3. hcc_reader    — MetaTrader 5 .hcc FX history (deferred to the FX phase)
  4. synthetic     — generators with controllable structure, for unit tests and
                     gate power/size validation
"""

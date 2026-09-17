"""
universe_acct0.py — Symbol universe for account acct0 (144 symbols, 12 industries).

To replace a ticker in this account's universe:
    python swap_symbols.py '{"OLD": "NEW"}'            # defaults to --account acct0
    python swap_symbols.py --account acct0 '{"OLD": "NEW"}'

Re-normalized 2026-09-17 to the price band M=$60 +/- 50% ($30-$90, max/min = 3.0x).

Why the band: the fill simulation now floors to whole shares (production submits qty=, never
notional=, and the stop-loss orders forbid fractional trading). With whole shares the price
decides how many distinct position sizes a symbol offers -- floor(0.60 * capital / price). In the
old universe that ranged from 100+ down to ONE inside a single industry (financials spanned
$25-$1,817, a 73x spread), so a modest position in an expensive name floored to zero while the
same intent in a cheap name executed. The band caps that spread at 3x.

Selection: maximise mean daily range (the strategy earns from intraday range, not drift), subject
to holding each industry's price mean near $60 and its mean/median ratio near 1.0. Every symbol
also cleared >= ~5y history and >= $10M/day median dollar volume -- illiquid names make limit
fills unrealistic. 185 screened candidates for 144 slots.

Maintenance: mean/median ratio is the drift detector (scale-free, so unlike an absolute price
floor it never goes stale). Watch > 1.20 / swap > 1.35 on the upside; watch < 0.96 / swap < 0.93
on the downside -- asymmetric because a symbol can rise without bound but only fall to zero.
"""

INDUSTRIES: dict[str, list[str]] = {
    # High-beta semiconductors & hardware
    'tech_hardware':          ['ICHR' ,'LASR' ,'UCTT' ,'SMCI' ,'AMKR' ,'AMBA' ,'COHU' ,'VECO' ,'IPGP' ,'KLIC' ,'ON'   ,'SWKS'],
    # High-beta cloud / AI software
    'tech_software_ai':       ['ZETA' ,'FIVN' ,'APPN' ,'VRNS' ,'TENB' ,'RNG'  ,'ESTC' ,'YOU'  ,'WK'   ,'QTWO' ,'NTNX' ,'ALRM'],
    # High-beta fintech + traditional finance
    'financials':             ['AFRM' ,'EZPW' ,'XYZ'  ,'MC'   ,'VIRT' ,'WAL'  ,'PFSI' ,'LNC'  ,'PYPL' ,'SYF'  ,'OMF'  ,'GBCI'],
    # EVs, autos, travel — already volatile
    'consumer_discretionary': ['PII'  ,'ASO'  ,'OLLI' ,'PVH'  ,'SAH'  ,'SHOO' ,'DAN'  ,'THO'  ,'APTV' ,'WYNN' ,'BKE'  ,'GM'],
    # Streaming, social, gig economy
    'consumer_services':      ['RBLX' ,'CBRL' ,'SHAK' ,'ZG'   ,'PLNT' ,'CHDN' ,'WH'   ,'GOLF' ,'TNL'  ,'MTCH' ,'NFLX' ,'BYD'],
    # Biotech / genomics
    'health_care':            ['TMDX' ,'ARWR' ,'CRSP' ,'CDNA' ,'IMVT' ,'CLDX' ,'ATRC' ,'VCEL' ,'CYTK' ,'PTCT' ,'ALKS' ,'LIVN'],
    # Airlines + industrials
    'industrials':            ['KTOS' ,'ALGT' ,'PRIM' ,'ATRO' ,'MRCY' ,'ALK'  ,'BLBD' ,'B'    ,'LUV'  ,'AIN'  ,'DAL'  ,'ZWS'],
    # High-beta lifestyle/consumer
    'consumer_staples':       ['FRPT' ,'SFM'  ,'UNFI' ,'YETI' ,'DAR'  ,'DECK' ,'LW'   ,'MGM'  ,'JJSF' ,'SPB'  ,'KR'   ,'TSN'],
    # Volatile E&P + services
    'energy':                 ['TDW'  ,'SM'   ,'OII'  ,'HP'   ,'WFRD' ,'MUR'  ,'WHD'  ,'MTDR' ,'AR'   ,'CRC'  ,'SLB'  ,'OXY'],
    # Clean energy / renewables
    'utilities':              ['SEDG' ,'ENPH' ,'CWEN' ,'HASI' ,'CWT'  ,'NWE'  ,'UGI'  ,'SR'   ,'OGS'  ,'NJR'  ,'SWX'  ,'POR'],
    # Homebuilders + proptech
    'real_estate':            ['LGIH' ,'Z'    ,'TREX' ,'SKY'  ,'PATK' ,'BCC'  ,'BZH'  ,'HNI'  ,'LEN'  ,'KBH'  ,'UFPI' ,'MAS'],
    # Volatile precious-metal miners
    'materials':              ['MP'   ,'SSRM' ,'AA'   ,'PAAS' ,'AGI'  ,'GFI'  ,'MEOH' ,'SQM'  ,'FCX'  ,'NGVT' ,'TECK' ,'ASH'],
}

ALL_SYMBOLS: list[str] = [sym for syms in INDUSTRIES.values() for sym in syms]
INDUSTRY_NAMES: list[str] = list(INDUSTRIES.keys())

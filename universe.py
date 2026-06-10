"""
universe.py — Single source of truth for the 144-symbol trading universe.

All files that reference the industry→symbol mapping import from here.
To replace a ticker, run:  python swap_symbols.py '{"OLD": "NEW"}'
"""

INDUSTRIES: dict[str, list[str]] = {
    # High-beta semiconductors & hardware
    'tech_hardware':          ['NVDA','AMD', 'MU',  'SMCI','MRVL','ON',  'AMAT','LRCX','KLAC','TSM', 'SWKS','MPWR'],
    # High-beta cloud / AI software
    'tech_software_ai':       ['PLTR','SNOW','DDOG','NET', 'CRWD','ZS',  'PANW','NOW', 'ADBE','CRM', 'FTNT','OKTA'],
    # High-beta fintech + traditional finance
    'financials':             ['XYZ', 'PYPL','AFRM','UPST','MELI','COIN','GS',  'SCHW','C',   'COF', 'BX',  'APO' ],
    # EVs, autos, travel — already volatile
    'consumer_discretionary': ['TSLA','RCL', 'XPEV','LI',  'APTV','GM',  'LEA', 'WYNN','BKNG','ABNB','UBER','LYFT'],
    # Streaming, social, gig economy
    'consumer_services':      ['NFLX','ROKU','SPOT','META','IAC', 'PINS','DASH','RBLX','TTWO','LYV', 'MTCH','WBD' ],
    # Biotech / genomics
    'health_care':            ['MRNA','BNTX','IMVT','CRSP','ARWR','MYGN','NTRA','INMD','HIMS','BEAM','ACAD','BMRN'],
    # Airlines + industrials
    'industrials':            ['BA',  'GE',  'CAT', 'DE',  'DAL', 'UAL', 'XPO', 'LUV', 'ALK', 'GNRC','BTU', 'STLD'],
    # High-beta lifestyle/consumer
    'consumer_staples':       ['CELH','SFM', 'ELF', 'LULU','DECK','YETI','NKE', 'CROX','DKNG','PENN','MGM', 'CZR' ],
    # Volatile E&P + services
    'energy':                 ['FANG','DVN', 'OXY', 'APA','AR',  'EQT', 'RRC', 'SM',  'SLB', 'COP', 'EOG', 'VLO' ],
    # Clean energy / renewables
    'utilities':              ['ENPH','FSLR','SEDG','CWEN','VST', 'BE',  'BEP', 'DQ',  'CSIQ','JKS', 'HASI','NRG' ],
    # Homebuilders + proptech
    'real_estate':            ['DHI', 'LEN', 'PHM', 'TOL', 'MTH', 'KBH', 'BZH', 'TMHC','LGIH','CSGP','Z',   'SKY' ],
    # Volatile precious-metal miners
    'materials':              ['NEM', 'AEM', 'FCX', 'SCCO','TECK','AA',  'SQM', 'WPM', 'AU',  'PAAS','GFI', 'CDE' ],
}

ALL_SYMBOLS: list[str] = [sym for syms in INDUSTRIES.values() for sym in syms]
INDUSTRY_NAMES: list[str] = list(INDUSTRIES.keys())

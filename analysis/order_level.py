"""Find the signal at the ATOMIC level: one order, one symbol, one day.

Everything tested so far aggregated 12 symbols into one industry number before any model saw it.
StockNN does not work that way -- it places 4 decisions per symbol per day across 144 symbols. If
the signal lives per-symbol, industry aggregation averages it away, which would explain why every
industry-level test came back null while the thing producing the data is profitable.

Reconstructed per symbol-day from cfeat (the orders StockNN actually placed) plus raw bars:
    buy_price  = low_t + buy_price_frac  * (high_t - low_t)
    sell_price = low_t + sell_price_frac * (high_t - low_t)
    buy fills  if next_low  <= buy_price      -> return (next_close - buy_price)/buy_price
    sell fills if next_high >= sell_price     -> gain vs holding = (sell_price - next_close)/next_close

That gives ~122,000 order-days with a known outcome, against 10,000 industry-days.
"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
import read_mt1_dataset as D

with open('/root/trading-ht/universe.json') as f: U=json.load(f)
inds=list(U)

def bars(s):
    with open(f'/root/trading/stock_data/{s}.json') as fh: d=json.load(fh)['days']
    return (np.array([b['open'] for b in d],float), np.array([b['high'] for b in d],float),
            np.array([b['low'] for b in d],float),  np.array([b['close'] for b in d],float))

for nm,path in (('run A','/root/abc3/mt1_dataset.bin'),('run B','/root/oos/mt1_dataset.bin')):
    ds=D.read(path) if hasattr(D,'read') else D.parse(path)
    day=np.asarray(ds['day']); cf=ds['cfeat'].astype(float); F=list(D.CN_FIELDS)
    jb,js,jq,jsq=F.index('buy_price_frac'),F.index('sell_all_price_frac'),F.index('buy_frac_avail'),F.index('sell_frac_held')
    rows=[]
    for k,ind in enumerate(inds):
        for j,s in enumerate(U[ind]):
            try: o,h,l,c = bars(s)
            except Exception: continue
            off=len(c)-1255
            for t,dd in enumerate(day):
                i=off+int(dd)
                if i<1 or i+1>=len(c): continue
                if day[t]<400: continue
                hi,lo,cl = h[i],l[i],c[i]
                if not (hi>lo>0): continue
                nl,nh,nc = l[i+1],h[i+1],c[i+1]
                bf=cf[t,k,jb*1 + j*len(F)] if False else cf[t,k,j*len(F)+jb]
                sf=cf[t,k,j*len(F)+js]
                bq=cf[t,k,j*len(F)+jq]; sq=cf[t,k,j*len(F)+jsq]
                bp_=lo+bf*(hi-lo); sp_=lo+sf*(hi-lo)
                buy_fill = (bq>1e-6) and (nl<=bp_) and bp_>0
                sell_fill= (sq>1e-6) and (nh>=sp_) and sp_>0
                rows.append((k,j,int(dd),bf,sf,bq,sq,
                             (nc-bp_)/bp_ if buy_fill else np.nan,
                             (sp_-nc)/nc  if sell_fill else np.nan,
                             1.0 if buy_fill else 0.0, 1.0 if sell_fill else 0.0,
                             (cl-lo)/(hi-lo), (hi-lo)/cl))
    R=np.array(rows,float)
    buyret=R[:,7]; selret=R[:,8]; bfill=R[:,9]; sfill=R[:,10]
    print(f'\n=== {nm}: order-level reconstruction, day >= 400 ===')
    print(f'  symbol-days: {len(R):,}    (industry-days were ~10,000)')
    print(f'  buy orders placed  {int((R[:,5]>1e-6).sum()):>8,}   of which filled '
          f'{int(np.nansum(bfill)):>8,} ({100*np.nansum(bfill)/max((R[:,5]>1e-6).sum(),1):.1f}%)')
    print(f'  sell orders placed {int((R[:,6]>1e-6).sum()):>8,}   of which filled '
          f'{int(np.nansum(sfill)):>8,} ({100*np.nansum(sfill)/max((R[:,6]>1e-6).sum(),1):.1f}%)')
    b=buyret[np.isfinite(buyret)]; s=selret[np.isfinite(selret)]
    print(f'\n  FILLED BUY  next-close return: mean {100*b.mean():+.3f}%  median {100*np.median(b):+.3f}%'
          f'  win rate {100*(b>0).mean():.1f}%  n={len(b):,}')
    print(f'  FILLED SELL vs holding:         mean {100*s.mean():+.3f}%  median {100*np.median(s):+.3f}%'
          f'  win rate {100*(s>0).mean():.1f}%  n={len(s):,}')
    print(f'\n  buy return by LIMIT PLACEMENT (buy_price_frac):')
    for lo_,hi_,lab in ((-0.01,0.01,'at the LOW  (0.0)'),(0.99,1.01,'at the HIGH (1.0)')):
        m=(R[:,3]>=lo_)&(R[:,3]<=hi_)&np.isfinite(buyret)
        if m.sum()>50:
            print(f'    {lab:<20} n={int(m.sum()):>7,}  mean {100*buyret[m].mean():+.3f}%  '
                  f'win {100*(buyret[m]>0).mean():.1f}%')
    print(f'  sell gain by LIMIT PLACEMENT (sell_all_price_frac):')
    for lo_,hi_,lab in ((-0.01,0.01,'at the LOW  (0.0)'),(0.99,1.01,'at the HIGH (1.0)')):
        m=(R[:,4]>=lo_)&(R[:,4]<=hi_)&np.isfinite(selret)
        if m.sum()>50:
            print(f'    {lab:<20} n={int(m.sum()):>7,}  mean {100*selret[m].mean():+.3f}%  '
                  f'win {100*(selret[m]>0).mean():.1f}%')
    np.save(f'/root/oos/orders_{nm.replace(" ","")}.npy', R)

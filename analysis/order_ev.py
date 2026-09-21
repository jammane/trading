"""Expected value per ORDER PLACED, not per fill -- the number that decides if placement matters.

A passive limit fills less often but at a better price. A limit that never fills costs the move it
missed. Comparing only FILLED orders is therefore biased toward passive placement; the honest
comparison charges the unfilled ones with their opportunity cost.

  filled buy    realised   (next_close - buy_price) / buy_price
  unfilled buy  forgone    (next_close - close_t) / close_t   <- what holding instead would have paid
  EV per order placed = fill_rate * realised + (1 - fill_rate) * 0   (capital stayed in cash)
  and separately, the opportunity-cost view where not filling costs the forgone move.
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
    return (np.array([b['high'] for b in d],float), np.array([b['low'] for b in d],float),
            np.array([b['close'] for b in d],float))

for nm,path in (('run A','/root/abc3/mt1_dataset.bin'),('run B','/root/oos/mt1_dataset.bin')):
    ds=D.read(path) if hasattr(D,'read') else D.parse(path)
    day=np.asarray(ds['day']); cf=ds['cfeat'].astype(float); F=list(D.CN_FIELDS); L=len(F)
    jb,js,jq,jsq=F.index('buy_price_frac'),F.index('sell_all_price_frac'),F.index('buy_frac_avail'),F.index('sell_frac_held')
    rec=[]
    for k,ind in enumerate(inds):
        for j,s in enumerate(U[ind]):
            try: h,l,c = bars(s)
            except Exception: continue
            off=len(c)-1255
            for t,dd in enumerate(day):
                if dd<400: continue
                i=off+int(dd)
                if i<1 or i+1>=len(c): continue
                hi,lo,cl=h[i],l[i],c[i]
                if not (hi>lo>0): continue
                nl,nh,nc=l[i+1],h[i+1],c[i+1]
                bf=cf[t,k,j*L+jb]; sf=cf[t,k,j*L+js]
                bq=cf[t,k,j*L+jq]; sq=cf[t,k,j*L+jsq]
                if bq>1e-6:
                    bp_=lo+bf*(hi-lo)
                    fill=(nl<=bp_) and bp_>0
                    real=(nc-bp_)/bp_ if fill else 0.0
                    forgone=(nc-cl)/cl
                    rec.append((0, 1.0 if bf>0.5 else 0.0, 1.0 if fill else 0.0, real, forgone))
                if sq>1e-6:
                    sp_=lo+sf*(hi-lo)
                    fill=(nh>=sp_) and sp_>0
                    real=(sp_-nc)/nc if fill else 0.0
                    forgone=0.0
                    # For a SELL a HIGH limit is PASSIVE (waits for a rally) and a LOW limit is
                    # AGGRESSIVE (sells into whatever is there). The buy convention is the
                    # opposite, and labelling both by frac>0.5 inverted the sell side.
                    rec.append((1, 0.0 if sf>0.5 else 1.0, 1.0 if fill else 0.0, real, forgone))
    R=np.array(rec,float)
    print(f'\n=== {nm}: expected value per ORDER PLACED, day >= 400 ===')
    print(f'  {"side":<6} {"placement":<12} {"placed":>8} {"fill%":>7} {"EV/order":>10} '
          f'{"EV incl. missed":>16} {"mean forgone":>13}')
    for side,slab in ((0,'BUY'),(1,'SELL')):
        for aggr,alab in ((0.0,'passive'),(1.0,'aggressive')):
            m=(R[:,0]==side)&(R[:,1]==aggr)
            if m.sum()<100: continue
            fr=R[m,2].mean(); ev=R[m,3].mean()
            miss=(1-R[m,2])*R[m,4]
            ev_opp=ev - miss.mean() if side==0 else ev
            print(f'  {slab:<6} {alab:<12} {int(m.sum()):>8,} {100*fr:>6.1f}% {100*ev:>+9.3f}% '
                  f'{100*ev_opp:>+15.3f}% {100*R[m,4].mean():>+12.3f}%')
    print('  EV/order      = fill_rate x realised return (unfilled = capital stayed in cash, 0)')
    print('  EV incl.missed= also charges an unfilled BUY the move it declined to participate in')

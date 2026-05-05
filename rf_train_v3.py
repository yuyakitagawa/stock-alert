
import requests, pandas as pd, numpy as np, time, os, io, joblib, json
from datetime import datetime, timedelta, date
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

FORECAST=63; RISE_THRESHOLD=15.0; DROP_THRESHOLD=15.0  # 絶対リターン閾値(%)
SAMPLE_INTERVAL=20; HISTORY_DAYS=1800
TRAIN_CUTOFF=date(2025,1,1); RANDOM_SEED=42; SEQ_DAYS=60
MIN_HISTORY=252+SEQ_DAYS+FORECAST+10
SAVE_DIR=os.path.expanduser("~/stock-alert")
HEADERS={"User-Agent":"Mozilla/5.0","Accept":"application/json"}

def get_nikkei_df(days=2200):
    """日経225の株価をDateインデックスのDataFrameで返す"""
    end_ts=int(datetime.now().timestamp())
    start_ts=int((datetime.now()-timedelta(days=days)).timestamp())
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/%5EN225?interval=1d&period1={start_ts}&period2={end_ts}"
    try:
        resp=requests.get(url,headers=HEADERS,timeout=15)
        data=resp.json()
        result=data.get("chart",{}).get("result",[])
        if not result: return None
        ts=result[0].get("timestamp",[])
        closes=result[0].get("indicators",{}).get("adjclose",[{}])[0].get("adjclose",[])
        idx=pd.to_datetime(ts,unit="s",utc=True).tz_convert("Asia/Tokyo")
        df=pd.DataFrame({"Close":closes},index=idx)
        df=df.dropna(); df.index=df.index.date
        return df
    except Exception: return None

def get_tse_stock_list():
    url="https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        resp=requests.get(url,headers=HEADERS,timeout=30)
        df=pd.read_excel(io.BytesIO(resp.content),dtype=str)
        df.columns=df.columns.str.strip()
        code_col=[c for c in df.columns if "コード" in c][0]
        name_col=[c for c in df.columns if "銘柄名" in c][0]
        result=df[[code_col,name_col]].copy()
        result.columns=["code","name"]
        result["code"]=result["code"].str.strip()
        result=result[result["code"].str.match(r"^[1-9]\d{3}$")]
        return result.sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
    except Exception as e:
        print(f"銘柄リスト取得失敗: {e}"); return None

def get_prices(code,days=HISTORY_DAYS):
    ticker=f"{code}.T"
    end_ts=int(datetime.now().timestamp())
    start_ts=int((datetime.now()-timedelta(days=days)).timestamp())
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&period1={start_ts}&period2={end_ts}"
    try:
        resp=requests.get(url,headers=HEADERS,timeout=15)
        if resp.status_code!=200: return None
        data=resp.json()
        result=data.get("chart",{}).get("result",[])
        if not result: return None
        ts=result[0].get("timestamp",[])
        closes=result[0].get("indicators",{}).get("adjclose",[{}])[0].get("adjclose",[])
        volumes=result[0].get("indicators",{}).get("quote",[{}])[0].get("volume",[])
        if not ts or not closes: return None
        idx=pd.to_datetime(ts,unit="s",utc=True).tz_convert("Asia/Tokyo")
        df=pd.DataFrame({"Close":closes,"Volume":volumes},index=idx).dropna()
        df.index=df.index.date
        return df
    except: return None

def compute_seq_features(seq):
    """60日リターン系列 → 7次元要約統計量"""
    s=np.array(seq,dtype=float)
    if len(s)<2: return [0.0]*7
    mean_s=s.mean(); std_s=s.std()
    ac=float(np.corrcoef(s[:-1],s[1:])[0,1]) if std_s>0 else 0.0
    if np.isnan(ac): ac=0.0
    skew=float(((s-mean_s)**3).mean()/(std_s**3+1e-10))
    max_r=float(s.max()); min_r=float(s.min())
    pos_ratio=float((s>0).mean())
    t=np.arange(len(s),dtype=float); slope=float(np.polyfit(t,s,1)[0])
    mid=len(s)//2; recent_vs_early=float(s[mid:].mean()-s[:mid].mean())
    return [ac,skew,max_r,min_r,pos_ratio,slope,recent_vs_early]

def calc_rsi(p,period=14):
    if len(p)<period+1: return 50.0
    d=np.diff(p[-(period+1):]); g=np.where(d>0,d,0).mean(); l=np.where(d<0,-d,0).mean()
    return 100.0 if l==0 else 100-100/(1+g/l)

def compute_feat(p, v=None, nk_rets=None):
    if len(p)<91 or p[-1]==0: return None
    c=p[-1]
    r5=(c-p[-6])/p[-6] if len(p)>=6 else 0
    r20=(c-p[-21])/p[-21] if len(p)>=21 else 0
    r60=(c-p[-61])/p[-61] if len(p)>=61 else 0
    r90=(c-p[-91])/p[-91]
    ma5=p[-5:].mean(); ma25=p[-25:].mean() if len(p)>=25 else p.mean()
    ma75=p[-75:].mean() if len(p)>=75 else p.mean()
    m525=ma5/ma25-1 if ma25>0 else 0; m2575=ma25/ma75-1 if ma75>0 else 0
    rsi=calc_rsi(p)
    v20=np.diff(p[-21:])/p[-21:-1] if len(p)>=21 else np.array([0])
    v60=np.diff(p[-61:])/p[-61:-1] if len(p)>=61 else np.array([0])
    vol20=v20.std()*np.sqrt(252)*100; vol60=v60.std()*np.sqrt(252)*100
    w=p[-252:] if len(p)>=252 else p; hi,lo=w.max(),w.min()
    pos52=(c-lo)/(hi-lo) if hi>lo else 0.5
    seq_raw=np.clip(np.diff(p[-(SEQ_DAYS+1):])/p[-(SEQ_DAYS+1):-1],-0.2,0.2) if len(p)>=SEQ_DAYS+1 else np.zeros(SEQ_DAYS)
    rhi=p[-60:].max() if len(p)>=60 else p.max(); ddown60=(c-rhi)/rhi
    hi52=p[-252:].max() if len(p)>=252 else p.max(); fhi52=(c-hi52)/hi52
    stk=0
    for j in range(1,min(21,len(p))):
        if p[-j]<p[-j-1]: stk+=1
        else: break
    dstreak=stk/20.0; maccel=r5-(r20/4)
    ma5a=p[-10:-5].mean() if len(p)>=10 else ma5; ma25a=p[-30:-5].mean() if len(p)>=30 else ma25
    cprev=ma5a/ma25a-1 if ma25a>0 else 0; mcdir=m525-cprev
    if v is not None and len(v)>=20:
        va=np.array([x if x is not None else np.nan for x in v],dtype=float)
        va5=np.nanmean(va[-5:]) if len(va)>=5 else 1
        va20=np.nanmean(va[-20:]) if len(va)>=20 else 1
        va60=np.nanmean(va[-60:]) if len(va)>=60 else va20
        vr520=va5/va20 if va20>0 else 1.0
        vr2060=va20/va60 if va60>0 else 1.0
        vsurge=va[-1]/va20 if va20>0 and not np.isnan(va[-1]) else 1.0
    else:
        vr520,vr2060,vsurge=1.0,1.0,1.0
    nk5  = nk_rets[0] if nk_rets is not None else 0.0
    nk20 = nk_rets[1] if nk_rets is not None else 0.0
    nk60 = nk_rets[2] if nk_rets is not None else 0.0
    feat=[r5,r20,r60,r90,m525,m2575,rsi,vol20,vol60,pos52,ddown60,fhi52,dstreak,maccel,mcdir,vr520,vr2060,vsurge,nk5,nk20,nk60]+compute_seq_features(seq_raw)
    return None if any(np.isnan(feat[:10])+np.isinf(feat[:10])) else feat

def generate_samples(df, nk_df=None):
    closes=df["Close"].values; dates=list(df.index); n=len(closes)
    volumes=df["Volume"].tolist() if "Volume" in df.columns else None
    samples=[]; start_i=max(252+SEQ_DAYS,90)
    nk_dates=list(nk_df.index) if nk_df is not None else []
    nk_closes=nk_df["Close"].values if nk_df is not None else np.array([])
    for i in range(start_i,n-FORECAST,SAMPLE_INTERVAL):
        v_slice=volumes[:i+1] if volumes is not None else None
        nk_rets=None; i0=None
        if nk_df is not None:
            d0=dates[i]
            i0=next((j for j,d in enumerate(nk_dates) if d>=d0),None)
            if i0 is not None:
                i5 =max(0,i0-5);  i20=max(0,i0-20); i60=max(0,i0-60)
                nk5 =(nk_closes[i0]-nk_closes[i5]) /nk_closes[i5]  if nk_closes[i5]!=0  else 0
                nk20=(nk_closes[i0]-nk_closes[i20])/nk_closes[i20] if nk_closes[i20]!=0 else 0
                nk60=(nk_closes[i0]-nk_closes[i60])/nk_closes[i60] if nk_closes[i60]!=0 else 0
                nk_rets=(nk5,nk20,nk60)
        feat=compute_feat(closes[:i+1], v_slice, nk_rets)
        if feat is None or closes[i]==0: continue
        chg=(closes[i+FORECAST]-closes[i])/closes[i]*100
        # ラベルは絶対リターン
        samples.append((dates[i],feat,int(chg>=RISE_THRESHOLD),int(chg<=-DROP_THRESHOLD)))
    return samples

def train_model(X_tr,y_tr,X_te,y_te,label):
    print(f"\n[学習] {label}モデル...")
    pos=y_tr.sum(); neg=len(y_tr)-pos; spw=neg/pos if pos>0 else 1.0
    print(f"  正例:{int(pos):,} 負例:{int(neg):,} spw:{spw:.2f}")
    m=XGBClassifier(n_estimators=800,max_depth=5,learning_rate=0.04,subsample=0.8,early_stopping_rounds=50,
        colsample_bytree=0.7,min_child_weight=8,scale_pos_weight=spw,
        eval_metric="auc",random_state=RANDOM_SEED,n_jobs=-1)
    m.fit(X_tr,y_tr,eval_set=[(X_te,y_te)],verbose=100)
    y_prob=m.predict_proba(X_te)[:,1]
    auc=roc_auc_score(y_te,y_prob)
    print(f"  ✅ テストAUC: {auc:.4f}")
    print(classification_report(y_te,(y_prob>=0.5).astype(int),target_names=["負例","正例"]))
    return m

def main():
    print("="*60)
    print("rf_train_v3: TSE全銘柄 × 5年 ウォークフォワード学習")
    print(f"分割境界: {TRAIN_CUTOFF} / サンプル間隔: {SAMPLE_INTERVAL}日")
    print("="*60)
    stock_list=get_tse_stock_list()
    if stock_list is None: return
    print(f"対象: {len(stock_list)}銘柄")
    print("\n日経225データ取得中...")
    nk_df=get_nikkei_df()
    if nk_df is not None: print(f"  日経225: {len(nk_df)}日分取得")
    else: print("  日経225取得失敗 → 絶対リターンで学習")
    print(f"\n株価取得中（30〜60分かかります）...")
    train_X,train_yr,train_yd=[],[],[]
    test_X,test_yr,test_yd=[],[],[]
    train_dates,test_dates=[],[]
    success=0
    for i,row in stock_list.iterrows():
        df=get_prices(row["code"])
        if df is None or len(df)<MIN_HISTORY:
            time.sleep(0.2); continue
        for (sd,feat,lr,ld) in generate_samples(df, nk_df):
            if sd<TRAIN_CUTOFF:
                train_X.append(feat); train_yr.append(lr); train_yd.append(ld); train_dates.append(sd)
            else:
                test_X.append(feat); test_yr.append(lr); test_yd.append(ld); test_dates.append(sd)
        success+=1
        if success%100==0:
            print(f"  [{success}銘柄] 学習:{len(train_X):,} テスト:{len(test_X):,}")
        time.sleep(0.25)
    print(f"\n完了: {success}銘柄 / 学習:{len(train_X):,} テスト:{len(test_X):,}")
    if len(train_X)<1000: print("ERROR: サンプル不足"); return
    X_tr=np.array(train_X); X_te=np.array(test_X)
    yr_tr=np.array(train_yr); yr_te=np.array(test_yr)
    yd_tr=np.array(train_yd); yd_te=np.array(test_yd)
    # Cross-sectional rank features (rank within each sample date)
    print("\nクロスセクショナルランク特徴量を計算中...")
    all_X=np.vstack([X_tr,X_te]); all_dates=np.array(train_dates+test_dates)
    RANK_FEAT=[0,1,2,6,7,9]  # ret5,ret20,ret60,rsi,vol20,pos52
    rank_mat=np.full((len(all_X),len(RANK_FEAT)),0.5)
    _,inv=np.unique(all_dates,return_inverse=True)
    for d_idx in range(inv.max()+1):
        g=np.where(inv==d_idx)[0]
        if len(g)<2: continue
        for j,fi in enumerate(RANK_FEAT):
            v=all_X[g,fi]; o=np.argsort(np.argsort(v)); rank_mat[g,j]=o/(len(g)-1)
    all_X_aug=np.hstack([all_X,rank_mat])
    n_tr=len(X_tr); X_tr=all_X_aug[:n_tr]; X_te=all_X_aug[n_tr:]
    print(f"  特徴量次元: {all_X.shape[1]} → {all_X_aug.shape[1]}")
    print(f"\n--- サンプル統計 ---")
    print(f"上昇ラベル: 学習 {yr_tr.sum():,}/{len(yr_tr):,} ({yr_tr.mean()*100:.1f}%)  テスト {yr_te.sum():,}/{len(yr_te):,} ({yr_te.mean()*100:.1f}%)")
    print(f"下落ラベル: 学習 {yd_tr.sum():,}/{len(yd_tr):,} ({yd_tr.mean()*100:.1f}%)  テスト {yd_te.sum():,}/{len(yd_te):,} ({yd_te.mean()*100:.1f}%)")
    rise=train_model(X_tr,yr_tr,X_te,yr_te,"上昇")
    drop=train_model(X_tr,yd_tr,X_te,yd_te,"下落")
    joblib.dump(rise,os.path.join(SAVE_DIR,"rf_model.pkl"))
    joblib.dump(drop,os.path.join(SAVE_DIR,"rf_drop_model.pkl"))
    # Save all samples (train+test) with dates for purged CV validation
    npz_path=os.path.join(SAVE_DIR,"training_data.npz")
    all_X=np.vstack([X_tr,X_te]); all_yr=np.concatenate([yr_tr,yr_te]); all_yd=np.concatenate([yd_tr,yd_te])
    all_dates=np.array([str(d) for d in train_dates+test_dates])
    np.savez_compressed(npz_path,X=all_X,yr=all_yr,yd=all_yd,dates=all_dates)
    rise_auc=roc_auc_score(yr_te,rise.predict_proba(X_te)[:,1])
    drop_auc=roc_auc_score(yd_te,drop.predict_proba(X_te)[:,1])
    with open(os.path.join(SAVE_DIR,"baseline_auc.json"),"w") as f:
        json.dump({"rise":float(rise_auc),"drop":float(drop_auc)},f)
    print(f"\nデータ保存: {npz_path}")
    print("\n保存完了 ✅  次: python3 ~/stock-alert/validation.py  →  python3 ~/stock-alert/rank_stocks.py")

main()

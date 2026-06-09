
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests, pandas as pd, numpy as np, time, io, joblib, json
from datetime import datetime, timedelta, date
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lib.utils import IsotonicCalibrated, extract_features, calc_rsi, add_cs_rank_features, get_sector_cached, get_market_index_df_cached
from lib.fundamentals import get_pit_fundamentals

FORECAST=63; RISE_THRESHOLD=15.0; DROP_THRESHOLD=15.0  # 63日(3ヶ月)±15%ラベル（設計通り）
ALPHA_THRESHOLD=8.0      # 相対ラベル: stock - nikkei >= +8% → alpha_rise=1
DROP_ALPHA_THRESHOLD=8.0 # 相対ラベル: stock - nikkei <= -8% → alpha_drop=1
# 4モデルアンサンブル:
#   net = (rise_abs - drop_abs) + (rise_rel - drop_rel)
#   絶対モデルが「どれが上がるか」を担当（AUC高い）
#   相対モデルが「日経超過を狙う」を担当（アルファ最適化）
SAMPLE_INTERVAL=63; HISTORY_DAYS=1800  # 63日ごとにサンプル（FORECASTに合わせる）
TRAIN_CUTOFF=date(2026,1,1); RANDOM_SEED=42; SEQ_DAYS=60
MIN_HISTORY=252+SEQ_DAYS+FORECAST+10
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.dirname(PROJECT_DIR)  # repo root (stock-alert/)
HEADERS={"User-Agent":"Mozilla/5.0","Accept":"application/json"}

# スクリーナーフィルター定数（screener.pyと同値に保つ）
_SC_MIN_MOM=8.0; _SC_MAX_MOM=30.0; _SC_MIN_MOM20=0.0
_SC_MIN_VOL=22.0; _SC_MAX_VOL=50.0; _SC_MIN_PRICE=300
_SC_MIN_VR=1.0; _SC_MIN_REL=0.05  # 出来高比 / 日経比相対強度
_SC_MIN_RSI=45.0; _SC_MAX_RSI=70.0

def _fetch_index_df(ticker_encoded, days=2200):
    """Yahoo Finance から市場指数の日次終値を date-indexed DataFrame で返す"""
    end_ts=int(datetime.now().timestamp())
    start_ts=int((datetime.now()-timedelta(days=days)).timestamp())
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_encoded}?interval=1d&period1={start_ts}&period2={end_ts}"
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

def get_nikkei_df(days=2200):
    return _fetch_index_df("%5EN225", days)   # 日経はDBキャッシュ不要（training直接使用）

def get_vix_df(days=2200):
    return get_market_index_df_cached("VIX",    "%5EVIX",    days)

def get_sp500_df(days=2200):
    return get_market_index_df_cached("SP500",  "%5EGSPC",   days)

def get_usdjpy_df(days=2200):
    return get_market_index_df_cached("USDJPY", "USDJPY%3DX", days)

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


def passes_screener_at(p, v_slice, nk_ret_3m):
    """i時点でv1スクリーナー条件を満たすかチェック（先読みバイアスなし）"""
    if len(p) < 64 or p[-1] < _SC_MIN_PRICE: return False
    mom3m = (p[-1]-p[-64])/p[-64]*100
    if not (_SC_MIN_MOM <= mom3m <= _SC_MAX_MOM): return False
    mom20 = (p[-1]-p[-21])/p[-21]*100 if len(p)>=21 else 0
    if mom20 < _SC_MIN_MOM20: return False
    dr = np.diff(p[-61:])/p[-61:-1] if len(p)>=61 else np.diff(p[-21:])/p[-21:-1]
    vol = dr.std()*np.sqrt(252)*100
    if not (_SC_MIN_VOL <= vol <= _SC_MAX_VOL): return False
    if np.polyfit(np.arange(len(p[-60:])), p[-60:], 1)[0] <= 0: return False
    if v_slice is not None and len(v_slice)>=60:
        va=np.array([x if x is not None else np.nan for x in v_slice],dtype=float)
        va20=np.nanmean(va[-20:]); va60=np.nanmean(va[-60:])
        if va60>0 and va20/va60 < _SC_MIN_VR: return False
    if nk_ret_3m is not None:
        rel=(p[-1]-p[-64])/p[-64] - nk_ret_3m
        if rel < _SC_MIN_REL: return False
    rsi = calc_rsi(p)
    if not (_SC_MIN_RSI <= rsi <= _SC_MAX_RSI): return False
    return True


def generate_samples(df, nk_df=None, screener_only=False, sample_code=None,
                     vix_map=None, sp500_dates=None, sp500_closes_arr=None,
                     usdjpy_dates=None, usdjpy_closes_arr=None):
    """vix_map: {date: float}, sp500_dates/usdjpy_dates: sorted list of date,
    sp500_closes_arr/usdjpy_closes_arr: np.array"""
    import bisect
    closes=df["Close"].values; dates=list(df.index); n=len(closes)
    volumes=df["Volume"].tolist() if "Volume" in df.columns else None
    samples=[]; start_i=max(252+SEQ_DAYS,90)
    nk_dates=list(nk_df.index) if nk_df is not None else []
    nk_closes=nk_df["Close"].values if nk_df is not None else np.array([])
    for i in range(start_i,n-FORECAST,SAMPLE_INTERVAL):
        v_slice=volumes[:i+1] if volumes is not None else None
        nk_rets=None; nk_ret_3m=None; i0=None
        if nk_df is not None:
            d0=dates[i]
            i0=next((j for j,d in enumerate(nk_dates) if d>=d0),None)
            if i0 is not None:
                i5 =max(0,i0-5);  i20=max(0,i0-20); i60=max(0,i0-60)
                nk5 =(nk_closes[i0]-nk_closes[i5]) /nk_closes[i5]  if nk_closes[i5]!=0  else 0
                nk20=(nk_closes[i0]-nk_closes[i20])/nk_closes[i20] if nk_closes[i20]!=0 else 0
                nk60=(nk_closes[i0]-nk_closes[i60])/nk_closes[i60] if nk_closes[i60]!=0 else 0
                nk_rets=(nk5,nk20,nk60)
                i63=max(0,i0-63)
                nk_ret_3m=(nk_closes[i0]-nk_closes[i63])/nk_closes[i63] if nk_closes[i63]!=0 else None
        if screener_only and not passes_screener_at(closes[:i+1], v_slice, nk_ret_3m):
            continue
        # VIX および SP500 のサンプル日時点の値を取得
        vix_val = us5_val = us20_val = None
        if vix_map:
            vix_val = vix_map.get(dates[i])
            if vix_val is None:
                # 直近の営業日の値を使用（祝日対応）
                for d_adj in range(1, 5):
                    from datetime import timedelta as _td
                    vix_val = vix_map.get(dates[i] - _td(days=d_adj))
                    if vix_val is not None:
                        break
        if sp500_dates and sp500_closes_arr is not None:
            sp_i = bisect.bisect_right(sp500_dates, dates[i]) - 1
            if sp_i >= 20:
                v_now = sp500_closes_arr[sp_i]
                v5    = sp500_closes_arr[max(0, sp_i - 5)]
                v20   = sp500_closes_arr[max(0, sp_i - 20)]
                us5_val  = (v_now - v5)  / v5  * 100 if v5  > 0 else None
                us20_val = (v_now - v20) / v20 * 100 if v20 > 0 else None
        # USD/JPY ベータ（60日ローリング）と5日リターン
        fx_beta_val = jpy5_val = None
        if usdjpy_dates and usdjpy_closes_arr is not None:
            fx_i = bisect.bisect_right(usdjpy_dates, dates[i]) - 1
            if fx_i >= 5:
                fx_now = usdjpy_closes_arr[fx_i]
                fx_5ago = usdjpy_closes_arr[max(0, fx_i - 5)]
                if fx_5ago > 0:
                    jpy5_val = (fx_now - fx_5ago) / fx_5ago * 100
            if fx_i >= 60 and i >= 60:
                stock_rets = np.diff(closes[i-60:i+1]) / closes[i-60:i]
                fx_rets    = np.diff(usdjpy_closes_arr[fx_i-60:fx_i+1]) / usdjpy_closes_arr[fx_i-60:fx_i]
                min_len = min(len(stock_rets), len(fx_rets))
                if min_len >= 20:
                    sr = stock_rets[:min_len]; fr = fx_rets[:min_len]
                    var_fx = np.var(fr)
                    if var_fx > 0:
                        fx_beta_val = float(np.cov(sr, fr)[0, 1] / var_fx)
        # point-in-timeファンダ（fundamentals_annual から、その日に既知だった値のみ）
        fund = None
        if sample_code is not None:
            pit = get_pit_fundamentals(sample_code, dates[i])
            if pit is not None:
                price_now = closes[i]
                eps, bps, dps = pit.get("eps"), pit.get("bps"), pit.get("dps")
                div_yield = (dps / price_now * 100) if dps and dps > 0 and price_now > 0 else None
                fund = {
                    "per":                 (price_now / eps) if eps and eps > 0 else None,
                    "pbr":                 (price_now / bps) if bps and bps > 0 else None,
                    "roe":                 pit.get("roe"),
                    "days_to_earnings":    pit.get("days_to_earnings"),
                    "days_since_div_ex":   pit.get("days_since_div_ex"),
                    "month":               dates[i].month,
                    "div_yield":           div_yield,
                    "eps_growth":          pit.get("eps_growth"),
                    "dps_growth":          pit.get("dps_growth"),
                    "vix":                 vix_val,
                    "us5":                 us5_val,
                    "us20":                us20_val,
                    # 新規IB特徴量
                    "fx_beta":             fx_beta_val,
                    "jpy5":                jpy5_val,
                    "eps_surprise":        pit.get("eps_surprise"),
                    "bps_growth":          pit.get("bps_growth"),
                    "piotroski":           pit.get("piotroski"),
                    "payout":              pit.get("payout"),
                    "accruals":            pit.get("accruals"),
                }
            else:
                # ファンダなし銘柄でもマクロ・FX特徴量は渡す
                fund = {
                    "vix":     vix_val,
                    "us5":     us5_val,
                    "us20":    us20_val,
                    "fx_beta": fx_beta_val,
                    "jpy5":    jpy5_val,
                }
        else:
            # sample_code未指定の場合もマクロ・FX特徴量は渡す
            fund = {
                "vix":     vix_val,
                "us5":     us5_val,
                "us20":    us20_val,
                "fx_beta": fx_beta_val,
                "jpy5":    jpy5_val,
            }
        feat=extract_features(closes[:i+1], v_slice, nk_rets, fundamentals=fund)
        if feat is None or closes[i]==0: continue
        chg=(closes[i+FORECAST]-closes[i])/closes[i]*100

        # 絶対ラベル（識別力重視）
        label_rise=int(chg>=RISE_THRESHOLD)   # +5%以上
        label_drop=int(chg<=-DROP_THRESHOLD)  # -5%以下

        # 相対ラベル（アルファ最適化）
        nk_ret_forecast=0.0
        if nk_df is not None and i0 is not None:
            i_fwd=next((j for j,d in enumerate(nk_dates) if d>=dates[i+FORECAST]),None)
            if i_fwd is not None and nk_closes[i0]!=0:
                nk_ret_forecast=(nk_closes[i_fwd]-nk_closes[i0])/nk_closes[i0]*100
        rel_chg=chg-nk_ret_forecast
        alpha_rise=int(rel_chg>=ALPHA_THRESHOLD)      # 日経+3%超
        alpha_drop=int(rel_chg<=-DROP_ALPHA_THRESHOLD) # 日経-3%以下

        samples.append((dates[i],feat,label_rise,label_drop,alpha_rise,alpha_drop))
    return samples

def train_model(X_tr,y_tr,X_te,y_te,X_cal,y_cal,label):
    print(f"\n[学習] {label}モデル...")
    pos=y_tr.sum(); neg=len(y_tr)-pos; spw=neg/pos if pos>0 else 1.0
    print(f"  正例:{int(pos):,} 負例:{int(neg):,} spw:{spw:.2f}")
    m=XGBClassifier(n_estimators=5000,max_depth=5,learning_rate=0.005,subsample=0.65,early_stopping_rounds=150,
        colsample_bytree=0.45,min_child_weight=60,reg_alpha=1.5,reg_lambda=5,gamma=0.5,scale_pos_weight=spw,
        eval_metric="auc",random_state=RANDOM_SEED,n_jobs=-1)
    m.fit(X_tr,y_tr,eval_set=[(X_te,y_te)],verbose=100)
    auc_raw=roc_auc_score(y_te,m.predict_proba(X_te)[:,1])
    print(f"  テストAUC（生）: {auc_raw:.4f}")
    iso=IsotonicRegression(out_of_bounds="clip")
    iso.fit(m.predict_proba(X_cal)[:,1],y_cal)
    cal_m=IsotonicCalibrated(m,iso)
    auc_cal=roc_auc_score(y_te,cal_m.predict_proba(X_te)[:,1])
    print(f"  ✅ テストAUC（キャリブレーション後）: {auc_cal:.4f}")
    print(classification_report(y_te,(cal_m.predict_proba(X_te)[:,1]>=0.5).astype(int),target_names=["負例","正例"]))
    return cal_m

def main():
    global TRAIN_CUTOFF
    import argparse as _ap
    _p=_ap.ArgumentParser(add_help=False)
    _p.add_argument("--screener-only",action="store_true",help="スクリーナー通過時点のサンプルのみ学習")
    _p.add_argument("--cutoff",type=str,default=None,help="学習cutoff日 YYYY-MM-DD（デフォルト: 2025-01-01）")
    _args,_=_p.parse_known_args()
    screener_only=_args.screener_only
    if _args.cutoff:
        TRAIN_CUTOFF = date.fromisoformat(_args.cutoff)
        print(f"カスタム cutoff: {TRAIN_CUTOFF}")

    print("="*60)
    print("rf_train_v3: TSE全銘柄 × 5年 ウォークフォワード学習")
    print(f"分割境界: {TRAIN_CUTOFF} / サンプル間隔: {SAMPLE_INTERVAL}日")
    if screener_only: print("【スクリーナー通過時点のみ学習モード】")
    print("="*60)
    stock_list=get_tse_stock_list()
    if stock_list is None: return
    print(f"対象: {len(stock_list)}銘柄")
    print("\n市場データ取得中...")
    nk_df=get_nikkei_df()
    if nk_df is not None: print(f"  日経225:    {len(nk_df)}日分取得")
    else: print("  日経225取得失敗 → 絶対リターンで学習")
    vix_df=get_vix_df()
    sp500_df=get_sp500_df()
    usdjpy_df=get_usdjpy_df()
    if vix_df is not None:    print(f"  VIX:        {len(vix_df)}日分取得（恐怖指数）")
    else:                     print("  VIX取得失敗 → 0.0で代替")
    if sp500_df is not None:  print(f"  S&P500:     {len(sp500_df)}日分取得（クロスアセット）")
    else:                     print("  S&P500取得失敗 → 0.0で代替")
    if usdjpy_df is not None: print(f"  USD/JPY:    {len(usdjpy_df)}日分取得（為替ベータ）")
    else:                     print("  USD/JPY取得失敗 → 0.0で代替")
    # date→値のマップを事前構築（generate_samples 内で高速ルックアップ）
    vix_map     = dict(zip(vix_df.index, vix_df["Close"]))      if vix_df    is not None else {}
    sp500_dates = sorted(sp500_df.index)                         if sp500_df  is not None else []
    sp500_closes_arr = (sp500_df.loc[sp500_dates, "Close"].values
                        if sp500_df is not None else np.array([]))
    usdjpy_dates = sorted(usdjpy_df.index)                       if usdjpy_df is not None else []
    usdjpy_closes_arr = (usdjpy_df.loc[usdjpy_dates, "Close"].values
                         if usdjpy_df is not None else np.array([]))
    # ファンダは fundamentals_annual から point-in-time で取得（事前に
    # tools/fetch_fundamentals_history.py で蓄積しておくこと）
    from lib.db import get_fundamentals_codes_count
    print(f"\nファンダDB: {get_fundamentals_codes_count()}銘柄分の年度別データを使用（point-in-time）")

    print(f"\n株価取得中（30〜60分かかります）...")
    train_X,train_yr,train_yd,train_ar,train_ad=[],[],[],[],[]
    test_X,test_yr,test_yd,test_ar,test_ad=[],[],[],[],[]
    train_dates,test_dates=[],[]
    train_sectors,test_sectors=[],[]   # セクター内相対モメンタムCS用
    success=0
    for i,row in stock_list.iterrows():
        code=str(row["code"])
        sector=get_sector_cached(code)   # JPX 33業種（プロセス内キャッシュ）
        df=get_prices(code)
        if df is None or len(df)<MIN_HISTORY:
            time.sleep(0.2); continue
        for (sd,feat,lr,ld,ar,ad) in generate_samples(df, nk_df, screener_only=screener_only,
                                                        sample_code=code,
                                                        vix_map=vix_map,
                                                        sp500_dates=sp500_dates,
                                                        sp500_closes_arr=sp500_closes_arr,
                                                        usdjpy_dates=usdjpy_dates,
                                                        usdjpy_closes_arr=usdjpy_closes_arr):
            if sd<TRAIN_CUTOFF:
                train_X.append(feat); train_yr.append(lr); train_yd.append(ld)
                train_ar.append(ar); train_ad.append(ad); train_dates.append(sd)
                train_sectors.append(sector)
            else:
                test_X.append(feat); test_yr.append(lr); test_yd.append(ld)
                test_ar.append(ar); test_ad.append(ad); test_dates.append(sd)
                test_sectors.append(sector)
        success+=1
        if success%100==0:
            print(f"  [{success}銘柄] 学習:{len(train_X):,} テスト:{len(test_X):,}")
        time.sleep(0.25)
    print(f"\n完了: {success}銘柄 / 学習:{len(train_X):,} テスト:{len(test_X):,}")
    if len(train_X)<1000: print("ERROR: サンプル不足"); return
    X_tr=np.array(train_X); X_te=np.array(test_X)
    yr_tr=np.array(train_yr); yr_te=np.array(test_yr)
    yd_tr=np.array(train_yd); yd_te=np.array(test_yd)
    ar_tr=np.array(train_ar); ar_te=np.array(test_ar)
    ad_tr=np.array(train_ad); ad_te=np.array(test_ad)
    # Cross-sectional rank features（6標準CS + 1セクター内相対モメンタム = 7 CS）
    print("\nクロスセクショナルランク特徴量を計算中...")
    all_X=np.vstack([X_tr,X_te]); all_dates=np.array(train_dates+test_dates)
    all_sectors=np.array(train_sectors+test_sectors)
    all_X_aug=add_cs_rank_features(all_X, dates=all_dates, sectors=all_sectors)
    n_tr=len(X_tr); X_tr=all_X_aug[:n_tr]; X_te=all_X_aug[n_tr:]
    print(f"  特徴量次元: {all_X.shape[1]} → {all_X_aug.shape[1]}")
    print(f"\n--- サンプル統計 ---")
    print(f"絶対上昇ラベル(≥+{RISE_THRESHOLD}%):  学習 {yr_tr.sum():,}/{len(yr_tr):,} ({yr_tr.mean()*100:.1f}%)  テスト {yr_te.sum():,}/{len(yr_te):,} ({yr_te.mean()*100:.1f}%)")
    print(f"絶対下落ラベル(≤-{DROP_THRESHOLD}%):  学習 {yd_tr.sum():,}/{len(yd_tr):,} ({yd_tr.mean()*100:.1f}%)  テスト {yd_te.sum():,}/{len(yd_te):,} ({yd_te.mean()*100:.1f}%)")
    print(f"相対上昇ラベル(α≥+{ALPHA_THRESHOLD}%): 学習 {ar_tr.sum():,}/{len(ar_tr):,} ({ar_tr.mean()*100:.1f}%)  テスト {ar_te.sum():,}/{len(ar_te):,} ({ar_te.mean()*100:.1f}%)")
    print(f"相対下落ラベル(α≤-{DROP_ALPHA_THRESHOLD}%): 学習 {ad_tr.sum():,}/{len(ad_tr):,} ({ad_tr.mean()*100:.1f}%)  テスト {ad_te.sum():,}/{len(ad_te):,} ({ad_te.mean()*100:.1f}%)")
    # キャリブレーション用: 学習データを日付順に並べ、最新20%をキャリブレーション用に分離
    sort_idx=np.argsort(np.array(train_dates))
    X_tr_s=X_tr[sort_idx]
    yr_s=yr_tr[sort_idx]; yd_s=yd_tr[sort_idx]
    ar_s=ar_tr[sort_idx]; ad_s=ad_tr[sort_idx]
    n_cal=max(500,int(len(X_tr_s)*0.2))
    X_tr_fit,X_cal=X_tr_s[:-n_cal],X_tr_s[-n_cal:]
    yr_fit,yr_cal=yr_s[:-n_cal],yr_s[-n_cal:]
    yd_fit,yd_cal=yd_s[:-n_cal],yd_s[-n_cal:]
    ar_fit,ar_cal=ar_s[:-n_cal],ar_s[-n_cal:]
    ad_fit,ad_cal=ad_s[:-n_cal],ad_s[-n_cal:]
    print(f"\nキャリブレーション分割: 学習{len(X_tr_fit):,} / キャリブレーション{len(X_cal):,} (最新20%)")
    rise =train_model(X_tr_fit,yr_fit, X_te,yr_te, X_cal,yr_cal, "絶対上昇")
    drop =train_model(X_tr_fit,yd_fit, X_te,yd_te, X_cal,yd_cal, "絶対下落")
    a_rise=train_model(X_tr_fit,ar_fit, X_te,ar_te, X_cal,ar_cal, "相対上昇(α)")
    a_drop=train_model(X_tr_fit,ad_fit, X_te,ad_te, X_cal,ad_cal, "相対下落(α)")
    cutoff_tag = f"_{TRAIN_CUTOFF.isoformat()}" if _args.cutoff else ""
    suffix="_screened" if screener_only else ""
    joblib.dump(rise,  os.path.join(SAVE_DIR,f"rf_model{suffix}{cutoff_tag}.pkl"))
    joblib.dump(drop,  os.path.join(SAVE_DIR,f"rf_drop_model{suffix}{cutoff_tag}.pkl"))
    joblib.dump(a_rise,os.path.join(SAVE_DIR,f"rf_alpha_model{suffix}{cutoff_tag}.pkl"))
    joblib.dump(a_drop,os.path.join(SAVE_DIR,f"rf_alpha_drop_model{suffix}{cutoff_tag}.pkl"))
    # Save all samples (train+test) with dates for purged CV validation
    npz_path=os.path.join(SAVE_DIR,"training_data.npz")
    all_X=np.vstack([X_tr,X_te]); all_yr=np.concatenate([yr_tr,yr_te]); all_yd=np.concatenate([yd_tr,yd_te])
    all_dates=np.array([str(d) for d in train_dates+test_dates])
    np.savez_compressed(npz_path,X=all_X,yr=all_yr,yd=all_yd,dates=all_dates)
    rise_auc  =roc_auc_score(yr_te,rise.predict_proba(X_te)[:,1])
    drop_auc  =roc_auc_score(yd_te,drop.predict_proba(X_te)[:,1])
    a_rise_auc=roc_auc_score(ar_te,a_rise.predict_proba(X_te)[:,1])
    a_drop_auc=roc_auc_score(ad_te,a_drop.predict_proba(X_te)[:,1])
    print(f"\n【AUC サマリー】")
    print(f"  絶対上昇: {rise_auc:.4f}  絶対下落: {drop_auc:.4f}")
    print(f"  相対上昇: {a_rise_auc:.4f}  相対下落: {a_drop_auc:.4f}")
    with open(os.path.join(SAVE_DIR,"baseline_auc.json"),"w") as f:
        json.dump({"rise":float(rise_auc),"drop":float(drop_auc),
                   "alpha_rise":float(a_rise_auc),"alpha_drop":float(a_drop_auc)},f)

    # C-1: 特徴量重要度を保存
    # 60次元: 53基本(32テクニカル+11ファンダ+4マクロ拡張+8新規IB) + 7クロスセクション
    feat_names = ["ret5","ret20","ret60","ret90","ma5_25","ma25_75","rsi","vol20","vol60","pos52",
                  "drawdown60","from_hi52","down_streak","momentum_accel","ma_cross_dir",
                  "vr520","vr2060","vsurge","nk5","nk20","nk60",
                  "ac","skew","max_ret","min_ret","pos_ratio","trend_slope","recent_vs_early",
                  "rel5","rel20","rel60","alpha_momentum",
                  "per_feat","pbr_feat","roe_feat","earn_feat",
                  "div_ex_feat","sin_month","cos_month","div_yield_f",
                  "eps_growth_f",
                  "dps_growth_f","vix_feat","us5_f","us20_f",
                  "amihud_f","fx_beta_f","jpy5_f",
                  "eps_surprise_f","bps_growth_f","piotroski_f","payout_f","accruals_f",
                  "cs_ret5","cs_ret20","cs_ret60","cs_rsi","cs_vol20","cs_pos52",
                  "cs_sector_ret60"]
    imp = {"rise": {n: float(v) for n, v in zip(feat_names, rise.model.feature_importances_)},
           "drop": {n: float(v) for n, v in zip(feat_names, drop.model.feature_importances_)}}
    with open(os.path.join(SAVE_DIR,"feature_importance.json"),"w") as f:
        json.dump(imp, f, indent=2, ensure_ascii=False)
    print("  特徴量重要度: feature_importance.json")

    # C-2: F1最大化閾値を保存（クラス不均衡対応）
    thresholds = {}
    for label, model, y_te_lbl in [("rise", rise, yr_te), ("drop", drop, yd_te)]:
        probs = model.predict_proba(X_te)[:, 1]
        pre, rec, thr = precision_recall_curve(y_te_lbl, probs)
        f1 = 2 * pre * rec / (pre + rec + 1e-10)
        best_thr = float(thr[f1[:-1].argmax()])
        thresholds[label] = best_thr
        print(f"  {label}モデル最適閾値: {best_thr:.3f} (F1={f1[:-1].max():.3f})")
    with open(os.path.join(SAVE_DIR,"optimal_thresholds.json"),"w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nデータ保存: {npz_path}")
    print("\n保存完了 ✅  次: python3 validation.py  →  python3 rank_stocks.py")

if __name__ == "__main__":
    main()

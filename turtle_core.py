"""
海龟交易分析系统 - 核心分析逻辑
==============================
运行方式：在终端执行 streamlit run app.py

海龟交易法则简介：
- 由Richard Dennis和William Eckhardt在1980年代创立
- 基于唐奇安通道突破的趋势跟踪系统
- 核心三要素：趋势识别、ATR头寸管理、金字塔加仓

依赖库：akshare, pandas, numpy, streamlit, plotly
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# 使用 curl_cffi 绕过东方财富的反爬 TLS 指纹检测
from curl_cffi import requests as curl_requests


def _fetch_spot_data():
    """
    获取全市场实时行情

    数据来源：
    1. 股票代码表：akshare stock_info_sh_name_code / sz_name_code
    2. 实时行情：Sina API（不受反爬限制），批量请求优化速度

    Sina 数据不含 PE/PB，这两项设为 "N/A"
    若需要基本面数据，可在个股详情页单独尝试获取。

    返回: DataFrame（代码, 名称, 最新价, 市盈率-动态, 市净率）
    """
    import time, math

    # ── 1. 获取代码列表 ──
    code_name_map = {}
    try:
        sh = ak.stock_info_sh_name_code()
        for _, r in sh.iterrows():
            c = str(r["证券代码"]).zfill(6)
            n = r.get("证券简称", c)
            code_name_map[c] = n
    except Exception:
        pass

    try:
        sz = ak.stock_info_sz_name_code()
        for _, r in sz.iterrows():
            c = str(r["A股代码"]).zfill(6)
            n = r.get("A股简称", c)
            code_name_map[c] = n
    except Exception:
        pass

    # 过滤 60/00 开头
    code_list = sorted(
        [c for c in code_name_map if c.startswith("60") or c.startswith("00")]
    )
    if not code_list:
        return pd.DataFrame()

    # ── 2. Sina 批量获取实时行情 ──
    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }

    def _batch_fetch(codes_batch):
        """一批请求获取多个股票实时价格"""
        sina_ids = [
            f"sh{c}" if c.startswith("60") else f"sz{c}" for c in codes_batch
        ]
        url = "https://hq.sinajs.cn/list=" + ",".join(sina_ids)
        r = curl_requests.get(url, headers=headers, impersonate="chrome124", timeout=15)
        r.encoding = "gbk"
        results = []
        for line in r.text.strip().split("\n"):
            if not line.startswith("var hq_str_"):
                continue
            sid = line.split("_")[2].split("=")[0]
            parts = line.split('"')[1].split(",")
            if len(parts) < 4:
                continue
            price_str = parts[3]
            if price_str == "" or price_str == "0.00":
                continue
            try:
                price = float(price_str)
            except ValueError:
                continue
            c = sid[2:]
            n = code_name_map.get(c, parts[0])
            results.append(
                {
                    "代码": c,
                    "名称": n,
                    "最新价": price,
                    "市盈率-动态": "N/A",
                    "市净率": "N/A",
                }
            )
        return results

    # 分批请求，每批 800 只
    batch_size = 800
    all_records = []
    for i in range(0, len(code_list), batch_size):
        batch = code_list[i : i + batch_size]
        all_records.extend(_batch_fetch(batch))
        if i + batch_size < len(code_list):
            time.sleep(0.3)

    df = pd.DataFrame(all_records)
    if df.empty:
        return df
    df["最新价"] = pd.to_numeric(df["最新价"], errors="coerce")
    df = df.dropna(subset=["最新价"])
    return df


def get_stock_pool(pool_type="沪深300"):
    """
    获取股票池

    参数:
    - pool_type: "沪深300" 或 "全市场"

    返回:
    - DataFrame，包含 代码、名称、最新价、市盈率-动态、市净率

    筛选规则:
    - 仅保留60（沪市主板）和00（深市主板）开头的股票
    - 排除30开头（创业板）和其他代码
    """
    all_df = _fetch_spot_data()
    # 仅保留60和00开头
    all_df = all_df[all_df["代码"].str.match(r"^(60|00)")].copy()

    if pool_type == "全市场":
        return all_df
    else:
        # 沪深300成分股
        cons = ak.index_stock_cons_csindex("000300")
        cons["成分券代码"] = cons["成分券代码"].astype(str).str.zfill(6)
        df = cons.merge(all_df, left_on="成分券代码", right_on="代码", how="inner")
        df = df.drop_duplicates(subset=["代码"])
        return df[["代码", "名称", "最新价", "市盈率-动态", "市净率"]].copy()


def _calc_atr(df, period):
    """
    计算ATR（平均真实波幅）- Wilder平滑法

    ATR衡量市场波动性，TR = max(H-L, |H-C|, |L-C|)
    海龟系统中ATR用于：
    1. 头寸规模：波动越大仓位越小
    2. 止损距离：入场价-2*ATR
    3. 加仓间距：每0.5*ATR加一次

    参数:
    - df: K线DataFrame（需含最高、最低、收盘列）
    - period: ATR周期（默认14）

    返回: ATR数组，长度与df相同
    """
    high = df["最高"].values.astype(float)
    low = df["最低"].values.astype(float)
    close = df["收盘"].values.astype(float)

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder平滑：初始SMA -> 递推
    atr = np.full(len(df), np.nan)
    atr[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, len(df)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _get_roe(codes):
    """
    批量获取个股ROE（净资产收益率）

    ROE = 净利润/净资产，反映股东权益的回报率
    用于基本面辅助筛选（并非海龟系统核心指标）

    参数:
    - codes: 股票代码列表

    返回: {code: roe_value} 字典，获取失败则为空字典
    """
    roe_dict = {}
    for code in codes:
        try:
            df = ak.stock_financial_abstract(code)
            if df is None or df.empty:
                continue
            # 查找净资产收益率
            if "净资产收益率" in df.iloc[:, 0].values:
                row = df[df.iloc[:, 0] == "净资产收益率"].iloc[0]
                val = row.iloc[1]
                if val != "-":
                    roe_dict[code] = float(val)
        except Exception:
            continue
    return roe_dict


def _get_hist_data(code, max_days=500):
    """
    获取个股历史日K线数据

    使用 stock_zh_a_daily（新浪数据源），绕开被屏蔽的 eastmoney 接口。
    注意：该函数内部使用了 py_mini_racer (V8)，不支持多线程。

    参数:
    - code: 6位股票代码
    - max_days: 最多保留多少条数据

    返回: DataFrame（日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额）或空DataFrame
    """
    try:
        code = str(code).zfill(6)
        prefix = "sh" if code.startswith("60") else "sz"
        df = ak.stock_zh_a_daily(f"{prefix}{code}")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "date": "日期",
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "volume": "成交量",
                "amount": "成交额",
            }
        )
        df = df.tail(max_days).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def _analyze_single_stock(code, name, pool_row, entry_period, exit_period, atr_period, risk_pct, equity, pe_col, pb_col):
    """分析单只股票，返回结果字典"""
    try:
        hist = _get_hist_data(code)
        if hist.empty or len(hist) < max(entry_period, atr_period) + 5:
            return None

        latest = hist.iloc[-1]
        current_price = float(latest["收盘"])

        atr_vals = _calc_atr(hist, atr_period)
        current_atr = atr_vals[-1]
        if np.isnan(current_atr) or current_atr <= 0:
            return None

        donchian_high = hist["最高"].rolling(window=entry_period).max().values
        donchian_low = hist["最低"].rolling(window=exit_period).min().values

        breakout = current_price > donchian_high[-2]
        stop_loss = current_price - 2 * current_atr
        position_units = int((equity * risk_pct) / current_atr / 100) * 100
        potential_breakout = donchian_high[-1]
        distance_pct = (potential_breakout - current_price) / current_price * 100

        add_targets = []
        if breakout:
            for i in range(1, 4):
                add_targets.append(round(current_price + i * 0.5 * current_atr, 2))

        return {
            "代码": code,
            "名称": name,
            "当前价": round(current_price, 2),
            "ATR": round(current_atr, 3),
            "突破信号": breakout,
            "建议手数": position_units,
            "止损价": round(stop_loss, 2),
            "潜在突破位(20日最高价)": round(potential_breakout, 2),
            "距离突破%": round(distance_pct, 2),
            "最新日期": str(latest["日期"]),
            "加仓机会": breakout and len(add_targets) > 0,
            "加仓目标价列表": str(add_targets),
            "PE": pool_row.get(pe_col, "-"),
            "PB": pool_row.get(pb_col, "-"),
        }
    except Exception:
        return None


def analyze_all_stocks(
    pool, entry_period=20, exit_period=10, atr_period=14, risk_pct=0.01, equity=1000000,
    progress_callback=None
):
    """
    海龟交易核心分析函数

    对股票池中每只股票逐一分析，应用海龟交易法则计算交易信号。

    参数:
    - pool: DataFrame（需含代码、名称）
    - entry_period: 入场通道周期（默认20日）
    - exit_period: 出场通道周期（默认10日）
    - atr_period: ATR计算周期（默认14日）
    - risk_pct: 每笔风险比例（默认1%）
    - equity: 账户总权益，单位元（默认100万）
    - progress_callback: 进度回调函数 fn(current, total, message)

    海龟规则:
    1. 入场 = 收盘价 > 前一日唐奇安通道上轨(entry_period日最高)
    2. 出场 = 收盘价 < 唐奇安通道下轨(exit_period日最低) 或 触发2*ATR止损
    3. 头寸 = (equity * risk_pct) / ATR，向下取整到100股
    4. 加仓 = 每突破0.5*ATR加一次，最多3次

    返回:
    - DataFrame: 代码,名称,当前价,ATR,突破信号(bool),建议手数,止损价,
      潜在突破位,距离突破%,最新日期,加仓机会(bool),加仓目标价列表,PE,PB,ROE
    """
    pe_col = "市盈率-动态" if "市盈率-动态" in pool.columns else "PE"
    pb_col = "市净率" if "市净率" in pool.columns else "PB"

    tasks = []
    for _, row in pool.iterrows():
        code = str(row["代码"]).zfill(6)
        name = row.get("名称", code)
        tasks.append((code, name, row.to_dict()))

    total = len(tasks)
    # 顺序执行（akshare 的 stock_zh_a_daily 内部使用 V8 引擎，不兼容多线程）
    results = []
    for idx, (code, name, row_dict) in enumerate(tasks):
        if progress_callback:
            progress_callback(idx + 1, total, f"分析 {name}({code})")

        r = _analyze_single_stock(
            code, name, row_dict,
            entry_period, exit_period, atr_period,
            risk_pct, equity, pe_col, pb_col,
        )
        if r is not None:
            results.append(r)

    if not results:
        return pd.DataFrame()

    df_result = pd.DataFrame(results)

    # 对有突破信号的股票获取ROE（用于基本面过滤）
    breakout_codes = df_result[df_result["突破信号"] == True]["代码"].tolist()
    if breakout_codes:
        roe_dict = _get_roe(breakout_codes)
        df_result["ROE"] = df_result["代码"].map(roe_dict)
    else:
        df_result["ROE"] = None

    return df_result


def get_stock_detail_data(
    code, entry_period=20, exit_period=10, atr_period=14, days=120
):
    """
    获取个股详细数据和K线技术指标，用于Streamlit详情展示

    返回:
    - info: dict 股票基本信息（名称、最新价、PE、PB、ROE）
    - plot_data: dict 包含:
        - kline: DataFrame（K线+唐奇安通道+ATR止损线）
        - breakout_dates: list（突破日期列表）
        - entry_period / exit_period
    """
    try:
        code = str(code).zfill(6)

        hist = _get_hist_data(code, max_days=days + max(entry_period, atr_period) + 20)
        if hist.empty:
            return None, None

        # --- 基本信息 ---
        info = {"名称": code, "最新价": float(hist.iloc[-1]["收盘"]), "PE": "-", "PB": "-", "ROE": "N/A"}
        try:
            spot = _fetch_spot_data()
            s = spot[spot["代码"] == code]
            if not s.empty:
                si = s.iloc[0]
                info["名称"] = si.get("名称", code)
                info["最新价"] = float(si.get("最新价", hist.iloc[-1]["收盘"]))
                info["PE"] = si.get("市盈率-动态", "-")
                info["PB"] = si.get("市净率", "-")
        except Exception:
            pass

        # ROE
        try:
            fin = ak.stock_financial_abstract(code)
            if fin is not None and not fin.empty:
                label_col = fin.columns[0]
                val_col = fin.columns[1]
                if "净资产收益率" in fin[label_col].values:
                    rv = fin[fin[label_col] == "净资产收益率"].iloc[0][val_col]
                    info["ROE"] = f"{rv}%" if rv != "-" else "N/A"
        except Exception:
            pass

        # --- 技术指标 ---
        hist["唐奇安上轨"] = hist["最高"].rolling(window=entry_period).max()
        hist["唐奇安下轨"] = hist["最低"].rolling(window=exit_period).min()

        atr_arr = _calc_atr(hist, atr_period)
        hist["ATR"] = atr_arr
        hist["ATR止损线"] = hist["收盘"] - 2 * atr_arr

        # 突破日期检测
        hist["突破检测"] = hist["收盘"] > hist["唐奇安上轨"].shift(1)
        breakout_dates = hist[hist["突破检测"] == True]["日期"].astype(str).tolist()

        return info, {
            "kline": hist,
            "breakout_dates": breakout_dates,
            "entry_period": entry_period,
            "exit_period": exit_period,
            "current_price": info["最新价"],
        }
    except Exception:
        return None, None

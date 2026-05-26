"""
海龟交易分析系统 - Streamlit 可视化界面
========================================
运行方式：在终端执行 streamlit run app.py

功能：
- 侧边栏参数调节（股票池、通道周期、ATR周期、风险比例、权益）
- 一键分析，展示结论摘要、突破信号表、接近突破表、全部股票表
- 单击展开个股详细分析卡片（含交互式K线图）
- 下载完整分析结果CSV

依赖库：akshare, pandas, numpy, streamlit, plotly
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import json, os, glob
from turtle_core import analyze_all_stocks, get_stock_pool, get_stock_detail_data

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)


# ── 历史记录管理 ─────────────────────────────────────────
def list_history():
    """列出历史分析记录，按时间倒序"""
    files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    records = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            meta["file"] = f
            records.append(meta)
        except Exception:
            continue
    records.sort(key=lambda x: x.get("time", ""), reverse=True)
    return records


def save_history(df, params):
    """保存此次分析结果到历史记录"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = datetime.now().strftime("%Y-%m-%d %H:%M")

    # CSV 数据文件
    csv_file = os.path.join(HISTORY_DIR, f"result_{ts}.csv")
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    # 元数据（参数 + 摘要）
    meta = {
        "time": ts,
        "label": label,
        "params": params,
        "csv_file": csv_file,
        "total_stocks": len(df),
        "breakout_count": int(df["突破信号"].sum()),
        "analysis_date": str(date.today()),
        "latest_data_date": str(df["最新日期"].iloc[0]) if "最新日期" in df.columns and len(df) > 0 else "",
    }
    meta_file = os.path.join(HISTORY_DIR, f"meta_{ts}.json")
    with open(meta_file, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    return meta


def load_history(meta):
    """加载历史分析结果"""
    try:
        df = pd.read_csv(meta["csv_file"])
        return df
    except Exception:
        return None

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(page_title="海龟交易分析看板", layout="wide")
st.title("🐢 海龟交易分析看板")

# ── Session state初始化（提升历史记录响应，消除rerun闪烁） ──
if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_source" not in st.session_state:
    st.session_state["df_source"] = None  # 历史记录meta，非history时为None
if "history_records" not in st.session_state:
    st.session_state["history_records"] = list_history()

# ── 侧边栏参数 ────────────────────────────────────────────
with st.sidebar:
    st.header("策略参数")

    pool_type = st.radio("股票池", ["沪深300", "全市场"], index=0)
    entry_period = st.number_input(
        "入场通道周期（唐奇安上轨N日最高）",
        min_value=5,
        max_value=120,
        value=20,
        step=1,
        help="收盘价突破过去N日最高价时入场，值越大过滤越严",
    )
    exit_period = st.number_input(
        "出场通道周期（唐奇安下轨N日最低）",
        min_value=5,
        max_value=120,
        value=10,
        step=1,
        help="收盘价跌破过去N日最低价时出场，值越小止损越快",
    )
    atr_period = st.number_input(
        "ATR计算周期",
        min_value=5,
        max_value=60,
        value=14,
        step=1,
        help="平均真实波幅的计算周期，越大越平滑",
    )
    risk_pct = st.number_input(
        "每笔风险比例",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="每笔交易承担的风险占总权益的比例，如0.01=1%",
    )
    equity = st.number_input(
        "账户总权益（元）",
        min_value=10000,
        max_value=100_000_000,
        value=1_000_000,
        step=10_000,
        format="%d",
    )

    st.divider()
    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    # ── 历史记录 ────────────────────────────────────────
    st.divider()
    st.header("📜 历史记录")
    history_records = st.session_state["history_records"]
    if history_records:
        hist_labels = [
            f"{r['label']}  |  {r['total_stocks']}只  |  {r['breakout_count']}个信号"
            for r in history_records
        ]
        selected_idx = st.selectbox(
            "选择历史分析查看",
            range(len(hist_labels)),
            format_func=lambda i: hist_labels[i],
            key="history_selector",
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("📂 加载", use_container_width=True):
                meta = history_records[selected_idx]
                csv_path = meta["csv_file"]
                df = load_history(meta)
                if df is not None and not df.empty:
                    st.session_state["df"] = df
                    st.session_state["df_source"] = meta
                    st.rerun()
                else:
                    st.error(f"❌ 文件加载失败：{csv_path}")
        with col_b:
            if st.button("🔄 刷新列表", use_container_width=True):
                st.session_state["history_records"] = list_history()
                st.rerun()
    else:
        st.caption("暂无历史记录，完成分析后自动保存")


# ── 缓存分析结果 ─────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def run_analysis_cached(pool_type, entry_p, exit_p, atr_p, risk, eq):
    """带缓存的分析入口（无进度显示）"""
    pool = get_stock_pool(pool_type)
    df = analyze_all_stocks(pool, entry_p, exit_p, atr_p, risk, eq)
    return pool, df

def run_analysis_with_progress(pool_type, entry_p, exit_p, atr_p, risk, eq):
    """带进度条的分析入口"""
    pool = get_stock_pool(pool_type)
    total = len(pool)
    progress_bar = st.progress(0, text="准备分析...")
    status_text = st.empty()

    def progress(current, total, msg):
        pct = min(current / total, 1.0)
        progress_bar.progress(pct, text=f"{msg} ({current}/{total})")
        status_text.text(f"分析进度: {current}/{total}")

    df = analyze_all_stocks(pool, entry_p, exit_p, atr_p, risk, eq, progress_callback=progress)
    progress_bar.empty()
    status_text.empty()
    return pool, df


# ── 绘图函数 ──────────────────────────────────────────────
def build_kline_chart(plot_data):
    """
    构建交互式K线图（plotly），包含：
    1. 日K线蜡烛图
    2. 唐奇安通道上轨/下轨
    3. ATR止损线
    4. 突破日期竖直虚线标注
    """
    kline = plot_data["kline"]
    entry_p = plot_data["entry_period"]
    exit_p = plot_data["exit_period"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
    )

    # ── 上子图：K线 + 通道线 ──
    fig.add_trace(
        go.Candlestick(
            x=kline["日期"],
            open=kline["开盘"],
            high=kline["最高"],
            low=kline["最低"],
            close=kline["收盘"],
            name="日K线",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        ),
        row=1,
        col=1,
    )

    # 唐奇安上轨
    fig.add_trace(
        go.Scatter(
            x=kline["日期"],
            y=kline["唐奇安上轨"],
            mode="lines",
            line=dict(color="#1565C0", width=1.5),
            name=f"唐奇安上轨({entry_p}日)",
        ),
        row=1,
        col=1,
    )

    # 唐奇安下轨
    fig.add_trace(
        go.Scatter(
            x=kline["日期"],
            y=kline["唐奇安下轨"],
            mode="lines",
            line=dict(color="#7B1FA2", width=1.5),
            name=f"唐奇安下轨({exit_p}日)",
        ),
        row=1,
        col=1,
    )

    # ATR止损线（仅显示最近的值以避免过度绘制）
    stop_line = kline[["日期", "ATR止损线"]].dropna(subset=["ATR止损线"])
    fig.add_trace(
        go.Scatter(
            x=stop_line["日期"],
            y=stop_line["ATR止损线"],
            mode="lines",
            line=dict(color="#D32F2F", width=1, dash="dash"),
            name="ATR止损线(收盘-2*ATR)",
        ),
        row=1,
        col=1,
    )

    # 突破日期竖线
    for bdate in plot_data.get("breakout_dates", []):
        fig.add_vline(
            x=bdate,
            line_dash="dash",
            line_color="#2E7D32",
            opacity=0.6,
        )

    # ── 下子图：成交量 ──
    colors = [
        "#ef5350" if c >= o else "#26a69a"
        for c, o in zip(kline["收盘"], kline["开盘"])
    ]
    fig.add_trace(
        go.Bar(
            x=kline["日期"],
            y=kline["成交量"],
            name="成交量",
            marker_color=colors,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    # ── 布局 ──
    fig.update_layout(
        title=dict(
            text="📊 K线图 — 海龟交易技术指标",
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(rangeslider_visible=False),
        yaxis=dict(title="价格"),
        yaxis2=dict(title="成交量"),
        height=650,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    fig.update_xaxes(matches="x", row=2, col=1)
    return fig


def highlight_rows(row):
    """突破信号为True的行显示为浅绿色背景"""
    if row.get("突破信号", False):
        return ["background-color: #d4edda"] * len(row)
    return [""] * len(row)


# ── 主逻辑 ────────────────────────────────────────────────

# 从 session_state 获取已有数据（跨 rerun 保持）
df = st.session_state["df"]
df_source = st.session_state["df_source"]

# 处理新分析请求
if analyze_btn:
    pool, df = run_analysis_with_progress(
        pool_type, entry_period, exit_period, atr_period, risk_pct, equity
    )
    st.success(f"✅ 分析完成！共分析 {len(pool)} 只股票，{len(df)} 只有效数据")
    if not df.empty:
        params = {
            "pool_type": pool_type,
            "entry_period": entry_period,
            "exit_period": exit_period,
            "atr_period": atr_period,
            "risk_pct": risk_pct,
            "equity": equity,
        }
        meta = save_history(df, params)
        st.session_state["history_records"] = list_history()  # 刷新历史列表
        st.caption(f"💾 已自动保存到历史记录（{meta['label']}）")
    # 存入 session_state，后续 rerun 不会丢失
    st.session_state["df"] = df
    st.session_state["df_source"] = None
    df_source = None  # 同步局部变量，避免误显示历史标记
elif df is None:
    # 首次加载引导
    st.info(
        "👈 请在左侧设置策略参数，然后点击 **「开始分析」** 运行海龟交易系统"
    )
    with st.expander("📖 什么是海龟交易法则？"):
        st.markdown(
            """
        - **起源**：1983年，Richard Dennis 和 William Eckhardt 用名为"海龟"的实验训练新手交易员，证明了交易技能可以后天习得
        - **核心**：趋势跟踪——不预测市场方向，只跟随已经出现的趋势
        - **入场**：价格突破过去20日最高价时买入（唐奇安通道突破）
        - **出场**：价格跌破过去10日最低价时卖出，或触发2倍ATR止损
        - **头寸管理**：用ATR衡量市场波动，波动大时减小仓位
        - **加仓**：每上涨0.5倍ATR加仓一次，最多加3次

        > 💡 本系统基于上述规则实时分析A股市场，提供交易参考，不构成投资建议
        """
        )
    st.stop()

# ── 显示历史记录加载提示 ──
if df_source is not None and isinstance(df_source, dict):
    st.info(f"📂 已加载历史分析：{df_source['label']}（分析日期：{df_source.get('analysis_date', '-')}，数据截止：{df_source.get('latest_data_date', '-')}）")

# ── 共用显示逻辑（新分析 & 历史记录） ────────────────────
if df.empty:
    st.warning("⚠️ 数据为空")
    st.stop()

# 分离信号
breakout_df = df[df["突破信号"] == True].copy()
near_breakout_df = df[
    (df["距离突破%"] > 0) & (df["距离突破%"] <= 5)
].copy()

# ── 1. 结论摘要 ──────────────────────────────────────────
st.header("📋 结论摘要")

total = len(df)
signal_count = len(breakout_df)

# 分析日期
data_date = df["最新日期"].iloc[0] if "最新日期" in df.columns and len(df) > 0 else "-"

# 基本面合理数量
pe_col = "PE"
pb_col = "PB"
roe_col = "ROE"
fundamental_ok = 0
if signal_count > 0 and roe_col in breakout_df.columns:
    valid = breakout_df[
        breakout_df[pe_col].apply(
            lambda x: isinstance(x, (int, float)) and x < 30
        )
        & breakout_df[pb_col].apply(
            lambda x: isinstance(x, (int, float)) and x < 3
        )
        & breakout_df[roe_col].apply(
            lambda x: isinstance(x, (int, float)) and x > 10
        )
    ]
    fundamental_ok = len(valid)

# 距离突破最近的前3只
near3 = df.nsmallest(3, "距离突破%")
near3_desc = "、".join(
    [
        f"{r['名称']}({r['距离突破%']:+.2f}%)"
        for _, r in near3.iterrows()
    ]
)

# 操作建议
if signal_count >= 5:
    advice = "🔥 出现多个优质突破信号，市场趋势较强，可积极参与"
elif signal_count >= 2:
    advice = "📈 出现少量突破信号，可轻仓参与，注意控制风险"
else:
    advice = "⏸️ 今日市场趋势偏弱，突破信号稀少，建议观望"

# 各板块KPI
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 分析股票总数", f"{total} 只")
with col2:
    st.metric("🚀 今日突破信号", f"{signal_count} 只")
with col3:
    st.metric(
        "✅ 基本面合理(P/E<30,PB<3,ROE>10%)",
        f"{fundamental_ok} 只" if fundamental_ok > 0 else "N/A",
    )
with col4:
    st.metric("📅 数据截止日期", data_date)

# 突破信号股票列表（明确显示代码和名称）
st.markdown("---")
if signal_count > 0:
    st.markdown("**🚀 触发突破信号的股票：**")
    breakout_list = "、".join(
        [f"**{r['名称']}**({r['代码']})" for _, r in breakout_df.iterrows()]
    )
    st.markdown(f"{breakout_list}")
    st.markdown("")

st.markdown("**距离突破最近的前3只：** " + near3_desc)
st.markdown("**操作建议：** " + advice)
st.markdown("---")

# ── 2. 今日突破信号表 ────────────────────────────────
st.header("🚀 今日突破信号")
if not breakout_df.empty:
    cols_show = [
        "代码",
        "名称",
        "当前价",
        "ATR",
        "建议手数",
        "止损价",
        "距离突破%",
        "加仓目标价列表",
    ]
    styled = (
        breakout_df[cols_show]
        .style.apply(highlight_rows, axis=1)
        .format(
            {
                "当前价": "{:.2f}",
                "ATR": "{:.3f}",
                "建议手数": "{:,}",
                "止损价": "{:.2f}",
                "距离突破%": "{:+.2f}%",
            }
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("暂无股票触发突破信号")

# ── 3. 接近突破表 ────────────────────────────────────
st.header("⏳ 接近突破（距离<5%）")
if not near_breakout_df.empty:
    cols_show = [
        "代码",
        "名称",
        "当前价",
        "ATR",
        "距离突破%",
        "潜在突破位(20日最高价)",
    ]
    styled = (
        near_breakout_df[cols_show]
        .style.apply(highlight_rows, axis=1)
        .format(
            {
                "当前价": "{:.2f}",
                "ATR": "{:.3f}",
                "距离突破%": "{:+.2f}%",
                "潜在突破位(20日最高价)": "{:.2f}",
            }
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("暂无接近突破的股票")

# ── 4. 全部股票表 ────────────────────────────────────
st.header("📄 全部股票分析结果")
all_cols = [
    "代码",
    "名称",
    "当前价",
    "ATR",
    "突破信号",
    "建议手数",
    "止损价",
    "潜在突破位(20日最高价)",
    "距离突破%",
    "加仓机会",
    "加仓目标价列表",
]
styled_all = (
    df[all_cols]
    .style.apply(highlight_rows, axis=1)
    .format(
        {
            "当前价": "{:.2f}",
            "ATR": "{:.3f}",
            "建议手数": "{:,}",
            "止损价": "{:.2f}",
            "潜在突破位(20日最高价)": "{:.2f}",
            "距离突破%": "{:+.2f}%",
        }
    )
)
st.dataframe(styled_all, use_container_width=True, hide_index=True)

# ── 5. 突破信号股票K线分析 ──────────────────────────
st.divider()
st.header("🔍 突破信号股票K线分析")

# 仅显示有突破信号的股票
breakout_stocks = df[df["突破信号"] == True]
if not breakout_stocks.empty:
    stock_options = breakout_stocks.apply(
        lambda r: f"{r['代码']} - {r['名称']}",
        axis=1,
    ).tolist()
    selected_label = st.selectbox(
        "选择突破股票查看K线分析", stock_options, key="stock_detail"
    )
    selected_code = selected_label.split(" - ")[0].strip()

    if selected_code:
        info, plot_data = get_stock_detail_data(
            selected_code, entry_period, exit_period, atr_period
        )
        if info is None:
            st.error(f"获取 {selected_code} 数据失败")
        else:
            # 从全部分析结果中获取该股票的数据
            stock_row = df[df["代码"] == selected_code]
            if stock_row.empty:
                st.warning("分析结果中未找到该股票数据")
                st.stop()
            sr = stock_row.iloc[0]

            # 基本信息卡片
            st.subheader(f"{info['名称']} ({selected_code})")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("当前价", f"{info['最新价']:.2f}")
            with c2:
                st.metric("ATR", f"{sr['ATR']:.3f}" if pd.notna(sr.get('ATR')) else "N/A")
            with c3:
                st.metric("建议手数", f"{sr['建议手数']:,}" if pd.notna(sr.get('建议手数')) else "N/A")
            with c4:
                st.metric("止损价", f"{sr['止损价']:.2f}" if pd.notna(sr.get('止损价')) else "N/A")
            pe_val = info.get("PE", "-")
            pb_val = info.get("PB", "-")
            roe_val = info.get("ROE", "N/A")
            with c5:
                st.metric("PE/PB/ROE", f"{pe_val}/{pb_val}/{roe_val}")
            st.markdown("---")

            # 入场与加仓信息
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**入场价：** {sr['当前价']:.2f}（唐奇安突破确认）")
                st.markdown(
                    f"**止损价：** {sr['止损价']:.2f} "
                    f"(入场价-2×ATR = {sr['当前价']:.2f} - 2×{sr['ATR']:.3f})"
                )
            with col_b:
                target_list = sr.get("加仓目标价列表", "[]")
                st.markdown("**加仓目标价：**")
                st.markdown(f"  - 第1次加仓(+0.5ATR): {sr['当前价'] + 0.5 * sr['ATR']:.2f}")
                st.markdown(f"  - 第2次加仓(+1.0ATR): {sr['当前价'] + 1.0 * sr['ATR']:.2f}")
                st.markdown(f"  - 第3次加仓(+1.5ATR): {sr['当前价'] + 1.5 * sr['ATR']:.2f}")

            # 交互式K线图
            if plot_data:
                fig = build_kline_chart(plot_data)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📖 图表说明"):
                    st.markdown(
                        """
                    - **红色/绿色蜡烛**：日K线（红涨绿跌）
                    - **蓝色线**：唐奇安通道上轨（入场参考线）
                    - **紫色线**：唐奇安通道下轨（出场参考线）
                    - **红色虚线**：ATR止损带（收盘价-2×ATR）
                    - **绿色虚线**：突破信号发生日期
                    - **下柱图**：成交量，颜色与涨跌一致
                    """
                    )
else:
    st.info("当前无突破信号股票，无K线图可显示")

# ── 6. 下载CSV ─────────────────────────────────────────
st.divider()
csv = df.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="📥 下载完整分析结果 CSV",
    data=csv,
    file_name=f"海龟交易分析_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
    use_container_width=True,
)

# ── 底部信息 ──────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ 免责声明：本系统仅供学习和研究参考，不构成任何投资建议。"
    "数据来源于 akshare 开源数据接口，可能存在延迟。"
    "投资有风险，入市需谨慎。"
)

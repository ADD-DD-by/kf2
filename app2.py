# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# ===================== 页面配置 =====================
st.set_page_config(page_title="问题层级处理时效分析", layout="wide")

st.markdown("""
<style>
    .main { background-color: #F5F6FA; }
    h1 { color: #2B3A67; text-align: center; padding: 0.5rem 0; border-bottom: 3px solid #5B8FF9; }
    h2, h3 { color: #2B3A67; margin-top: 1.2rem; }
    div.stButton > button:first-child {
        background-color: #5B8FF9; color: white; border: none; border-radius: 8px;
        padding: 0.4rem 1.0rem;
    }
    div.stButton > button:hover { background-color: #3A6CE5; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("💬 问题层级处理时效分析")

# ===================== 工具函数 =====================
NULL_LIKE_REGEX = {r"^[-‐-‒–—―−]+$": None, r"^(null|none|nan|NaN|NA)$": None, r"^\s*$": None}

def clean_numeric(s):
    s = s.astype(str).str.strip().replace(NULL_LIKE_REGEX, regex=True).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def safe_quantile(s, q=0.9):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.quantile(q) if len(s) > 0 else np.nan

def detect_created_col(df):
    candidates = [c for c in df.columns if "ticket_created" in c.lower() or "创建时间" in c]
    return candidates[0] if candidates else None

def ensure_time_month(df):
    created_col = detect_created_col(df)
    if created_col is None:
        st.error("❌ 未找到创建时间列（应包含 ticket_created 或 创建时间）")
        st.stop()
    df["ticket_created_datetime"] = pd.to_datetime(df[created_col], errors="coerce")
    df["month"] = df["ticket_created_datetime"].dt.to_period("M").astype(str)
    return df

def basic_clean(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace(NULL_LIKE_REGEX, regex=True)
    for col in ["处理时长", "评分", "message_count"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    return df

def group_metrics(df, level_cols, extra_dims):
    group_cols = extra_dims + level_cols
    df_valid = df.dropna(subset=["处理时长", "评分"])
    if df_valid.empty:
        return pd.DataFrame()
    grouped = (df_valid.groupby(group_cols, as_index=False)
               .agg(
                   回复次数_P90=("message_count", safe_quantile),
                   处理时长_P90=("处理时长", safe_quantile),
                   满意度_4_5占比=("评分", lambda x: (x >= 4).sum() / len(x) if len(x) > 0 else np.nan),
                   样本量=("评分", "count")
               ))
    sort_cols = [c for c in ["month", "business_line", "ticket_channel", "site_code"] if c in grouped.columns]
    return grouped.sort_values(sort_cols + level_cols)

def export_sheets(buff, sheets, filters_text):
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        pd.DataFrame({"筛选条件": [filters_text]}).to_excel(writer, index=False, sheet_name="筛选说明")
        for name, df in sheets.items():
            if not df.empty:
                df.to_excel(writer, index=False, sheet_name=name)
    buff.seek(0)

# ===================== 文件上传 =====================
uploaded = st.file_uploader("📂 上传一个或多个文件（Excel / CSV）", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded:
    dfs = []
    for f in uploaded:
        try:
            df = pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
            df = df.dropna(how="all").reset_index(drop=True)
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ 文件 {f.name} 读取失败：{e}")
    if not dfs:
        st.error("❌ 没有成功读取的文件")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ 已加载并合并 {len(dfs)} 个文件，共 {len(df)} 行数据。")
    st.dataframe(df.head(10), use_container_width=True)

    # ============= 数据清洗 =============
    df = ensure_time_month(df)
    df = basic_clean(df)
    for col in ["class_one", "class_two", "business_line", "ticket_channel", "site_code"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ============= 侧边栏筛选条件 =============
    st.sidebar.header("🔎 数据筛选条件")

    min_date, max_date = df["ticket_created_datetime"].min(), df["ticket_created_datetime"].max()
    start_date, end_date = st.sidebar.date_input(
        "选择时间范围",
        value=(min_date.date() if min_date else datetime.today().date(),
               max_date.date() if max_date else datetime.today().date())
    )

    month_sel = st.sidebar.multiselect("月份", sorted(df["month"].dropna().unique()))
    bl_sel = st.sidebar.multiselect("业务线", sorted(df["business_line"].dropna().unique()) if "business_line" in df.columns else [])
    ch_sel = st.sidebar.multiselect("渠道", sorted(df["ticket_channel"].dropna().unique()) if "ticket_channel" in df.columns else [])
    site_sel = st.sidebar.multiselect("国家", sorted(df["site_code"].dropna().unique()) if "site_code" in df.columns else [])

    df_f = df.copy()
    if start_date and end_date:
        df_f = df_f[
            (df_f["ticket_created_datetime"] >= pd.to_datetime(start_date)) &
            (df_f["ticket_created_datetime"] <= pd.to_datetime(end_date))
        ]
    if month_sel:
        df_f = df_f[df_f["month"].isin(month_sel)]
    if bl_sel:
        df_f = df_f[df_f["business_line"].isin(bl_sel)]
    if ch_sel:
        df_f = df_f[df_f["ticket_channel"].isin(ch_sel)]
    if site_sel:
        df_f = df_f[df_f["site_code"].isin(site_sel)]

    extra_dims = [c for c in ["month", "business_line", "ticket_channel", "site_code"] if c in df_f.columns]

    # ============= 指标计算 =============
    lvl1 = group_metrics(df_f, ["class_one"], extra_dims)
    lvl2 = group_metrics(df_f, ["class_one", "class_two"], extra_dims)

    st.header("📑 指标汇总结果")
    tab1, tab2 = st.tabs(["一级问题", "二级问题"])
    tab1.dataframe(lvl1, use_container_width=True)
    tab2.dataframe(lvl2, use_container_width=True)

    # ============= 柱+折线图 =============
    st.header("📊 问题类型对比图（柱=回复/时效，线=满意度）")
    level_choice = st.selectbox("选择问题层级", ["一级问题", "二级问题"], index=0)
    cur_df = lvl1 if level_choice == "一级问题" else lvl2

    if not cur_df.empty:
        x_col = "class_one" if level_choice == "一级问题" else "class_two"
        cur_df = cur_df.dropna(subset=["回复次数_P90", "处理时长_P90", "满意度_4_5占比"])

        metrics = ["回复次数_P90", "处理时长_P90", "满意度_4_5占比"]
        df_plot = cur_df.copy()
        for m in metrics:
            df_plot[m] = pd.to_numeric(df_plot[m], errors="coerce")
            if df_plot[m].max() != df_plot[m].min():
                df_plot[m + "_norm"] = (df_plot[m] - df_plot[m].min()) / (df_plot[m].max() - df_plot[m].min())
            else:
                df_plot[m + "_norm"] = df_plot[m]

        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
        df_plot = df_plot.groupby(x_col, as_index=False)[numeric_cols].mean()

        problem_choices = sorted(df_plot[x_col].unique())
        selected_problems = st.multiselect(f"选择要显示的{level_choice}", problem_choices, default=problem_choices[:15])
        if selected_problems:
            df_plot = df_plot[df_plot[x_col].isin(selected_problems)]

        bar_df = df_plot.melt(id_vars=[x_col], value_vars=["回复次数_P90_norm", "处理时长_P90_norm"],
                              var_name="指标", value_name="标准化数值")
        bar_df["指标"] = bar_df["指标"].replace({
            "回复次数_P90_norm": "回复次数P90",
            "处理时长_P90_norm": "处理时长P90"
        })

        fig = go.Figure()
        for metric, color in zip(["回复次数P90", "处理时长P90"], ["#5B8FF9", "#5AD8A6"]):
            data = bar_df[bar_df["指标"] == metric]
            fig.add_trace(go.Bar(
                x=data[x_col], y=data["标准化数值"], name=metric,
                marker_color=color, text=[f"{v:.2f}" for v in data["标准化数值"]],
                textposition="outside"
            ))

        fig.add_trace(go.Scatter(
            x=df_plot[x_col], y=df_plot["满意度_4_5占比_norm"],
            name="满意度(4/5占比)", mode="lines+markers+text",
            line=dict(color="#F6BD16", width=3),
            marker=dict(size=8),
            text=[f"{v:.2f}" for v in df_plot["满意度_4_5占比_norm"]],
            textposition="top center"
        ))

        fig.update_layout(
            title=f"{level_choice}：三指标对比（柱=回复/时效，线=满意度）",
            barmode="group", xaxis_title="问题类型", yaxis_title="标准化数值(0~1)",
            xaxis_tickangle=-30, plot_bgcolor="white",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============= 💬 气泡图 =============
    st.header("💬 回复次数/处理时长 与 满意度关系（气泡图）")
    if not cur_df.empty:
        scatter_choice = st.radio("选择横轴指标", ["处理时长_P90", "回复次数_P90"], horizontal=True)
        x_col_name = scatter_choice
        y_col_name = "满意度_4_5占比"
        size_col = "样本量"

        problem_field = "class_one" if level_choice == "一级问题" else "class_two"
        df_scatter = cur_df.copy().dropna(subset=[x_col_name, y_col_name, size_col])
        df_scatter["样本量_scaled"] = (df_scatter[size_col] / df_scatter[size_col].max()) * 80 + 10

        if not df_scatter.empty:
            fig_scatter = go.Figure()
            for pb in sorted(df_scatter[problem_field].unique()):
                data = df_scatter[df_scatter[problem_field] == pb]
                fig_scatter.add_trace(go.Scatter(
                    x=data[x_col_name],
                    y=data[y_col_name],
                    mode="markers+text",
                    name=str(pb),
                    text=[pb] * len(data),
                    textposition="top center",
                    marker=dict(
                        size=data["样本量_scaled"],
                        color=data[x_col_name],
                        colorscale="YlOrRd",
                        showscale=False,
                        line=dict(width=1, color="gray"),
                        opacity=0.85
                    ),
                    hovertemplate=(
                        f"{problem_field}: %{{text}}<br>"
                        f"{x_col_name}: %{{x:.2f}}<br>"
                        f"{y_col_name}: %{{y:.2f}}<br>"
                        f"样本量: %{{marker.size:.0f}}<extra></extra>"
                    )
                ))

            # 拟合趋势线
            if len(df_scatter) > 2:
                z = np.polyfit(df_scatter[x_col_name], df_scatter[y_col_name], 1)
                p = np.poly1d(z)
                fig_scatter.add_trace(go.Scatter(
                    x=df_scatter[x_col_name],
                    y=p(df_scatter[x_col_name]),
                    mode="lines",
                    line=dict(color="gray", dash="dot"),
                    name="趋势线"
                ))

            fig_scatter.update_layout(
                title=f"{level_choice}：{x_col_name} 与 {y_col_name} 的关系（按问题类型）",
                xaxis_title=x_col_name,
                yaxis_title=y_col_name,
                plot_bgcolor="white",
                height=650,
                title_x=0.5,
                title_font=dict(size=20, color="#2B3A67"),
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ============= 📈 月度趋势图 =============
    st.header("📈 不同问题下 处理时长 与 回复次数 的月度趋势")
    if "month" in df_f.columns:
        trend_level = st.radio("选择层级查看趋势", ["一级问题", "二级问题"], horizontal=True)
        df_trend = lvl1 if trend_level == "一级问题" else lvl2
        df_trend = df_trend.dropna(subset=["month", "处理时长_P90", "回复次数_P90"])
        if not df_trend.empty:
            problem_field = "class_one" if trend_level == "一级问题" else "class_two"
            problem_sel = st.multiselect(f"选择要展示的{trend_level}", sorted(df_trend[problem_field].unique()), default=sorted(df_trend[problem_field].unique())[:5])
            df_trend = df_trend[df_trend[problem_field].isin(problem_sel)]

            fig_trend = go.Figure()
            for pb in problem_sel:
                data = df_trend[df_trend[problem_field] == pb]
                fig_trend.add_trace(go.Scatter(
                    x=data["month"], y=data["处理时长_P90"],
                    name=f"{pb}-处理时长", mode="lines+markers",
                    line=dict(width=2), marker=dict(size=6)
                ))
                fig_trend.add_trace(go.Scatter(
                    x=data["month"], y=data["回复次数_P90"],
                    name=f"{pb}-回复次数", mode="lines+markers",
                    line=dict(dash="dot", width=2), marker=dict(size=6)
                ))

            fig_trend.update_layout(
                title=f"{trend_level}：处理时长 与 回复次数 月度趋势",
                xaxis_title="月份",
                yaxis_title="数值",
                plot_bgcolor="white",
                height=650,
                title_x=0.5,
                title_font=dict(size=20, color="#2B3A67"),
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # ============= 🏆 Top5 榜单 =============
    st.header("🏆 Top5 榜单")
    x_col = "class_one"
    df_rank = lvl1.groupby(x_col, as_index=False).agg({
        "处理时长_P90": "mean", "满意度_4_5占比": "mean", "样本量": "sum"
    })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⏱️ 处理时长最慢 Top5")
        if not df_rank.empty:
            top5_slow = df_rank.sort_values("处理时长_P90", ascending=False).head(5)
            st.dataframe(top5_slow, use_container_width=True)
    with col2:
        st.subheader("😞 满意度最低 Top5")
        if not df_rank.empty:
            top5_bad = df_rank.sort_values("满意度_4_5占比", ascending=True).head(5)
            st.dataframe(top5_bad, use_container_width=True)

    # ============= 🌍 热力图分析（稳定版） =============
    st.header("🌍 维度交叉热力图（满意度 or 时效）")
    if not df_f.empty:
        st.markdown("展示不同维度组合下的关键指标表现，可用于横向比较渠道、国家或业务线。")
        x_dim = st.selectbox("选择 X 轴维度", ["business_line", "ticket_channel", "site_code"], index=0)
        y_dim = st.selectbox("选择 Y 轴维度", ["ticket_channel", "site_code", "business_line"], index=1)
        metric_sel = st.radio("选择指标", ["满意度_4_5占比", "处理时长_P90", "回复次数_P90"], horizontal=True)
        if x_dim == y_dim:
            st.warning("⚠️ X 轴与 Y 轴不能相同。")
        else:
            df_hm = group_metrics(df_f.copy(), [], [x_dim, y_dim]).pivot(index=y_dim, columns=x_dim, values=metric_sel)
            if not df_hm.empty:
                x_vals, y_vals = df_hm.columns.tolist(), df_hm.index.tolist()
                z_vals = df_hm.values
                z_text = pd.DataFrame(z_vals, index=y_vals, columns=x_vals).round(2).astype(str).values
                fig_hm = go.Figure(data=go.Heatmap(
                    z=z_vals, x=x_vals, y=y_vals, colorscale="YlGnBu",
                    colorbar_title=str(metric_sel),
                    hovertemplate=f"{x_dim}: %{{x}}<br>{y_dim}: %{{y}}<br>{metric_sel}: %{{z:.3f}}<extra></extra>",
                    text=z_text, texttemplate="%{text}"
                ))
                fig_hm.update_layout(
                    title=f"{metric_sel} - {x_dim} × {y_dim} 热力图",
                    title_x=0.5, title_font=dict(size=20, color="#2B3A67"),
                    xaxis_title=x_dim, yaxis_title=y_dim,
                    xaxis_tickangle=-30, xaxis_tickfont=dict(size=14, color="#2B3A67"),
                    yaxis_tickfont=dict(size=14, color="#2B3A67"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=700, margin=dict(l=80, r=80, t=80, b=80)
                )
                st.plotly_chart(fig_hm, use_container_width=True)
    # ============= 📤 导出分析报告 =============
    st.header("📤 导出分析报告")
    st.markdown("将当前所有筛选条件与分析结果导出为 Excel 文件。")

    filters_text = f"时间范围: {start_date} 至 {end_date}; " \
                   f"月份: {', '.join(month_sel) if month_sel else '全部'}; " \
                   f"业务线: {', '.join(bl_sel) if bl_sel else '全部'}; " \
                   f"渠道: {', '.join(ch_sel) if ch_sel else '全部'}; " \
                   f"国家: {', '.join(site_sel) if site_sel else '全部'}"

    # 热力图数据（最后一次选择）
    try:
        df_heatmap_export = df_hm.reset_index()
    except Exception:
        df_heatmap_export = pd.DataFrame()

    sheets_dict = {
        "一级问题汇总": lvl1,
        "二级问题汇总": lvl2,
        f"{level_choice}气泡图数据": cur_df,
        f"{trend_level}月趋势数据": lvl1 if trend_level == "一级问题" else lvl2,
        "热力图透视表": df_heatmap_export
    }

    export_buffer = BytesIO()
    export_sheets(export_buffer, sheets_dict, filters_text)

    st.download_button(
        label="📥 点击下载 Excel 报告",
        data=export_buffer,
        file_name=f"问题层级分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

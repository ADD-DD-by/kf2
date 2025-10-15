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
    h1 { color: #2B3A67; text-align: center; padding: 0.5rem 1rem; border-bottom: 3px solid #5B8FF9; }
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

def clean_numeric(s: pd.Series) -> pd.Series:
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
            if isinstance(df, pd.DataFrame) and not df.empty:
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
    default_start = (min_date.date() if pd.notna(min_date) else datetime.today().date())
    default_end   = (max_date.date() if pd.notna(max_date) else datetime.today().date())
    start_date, end_date = st.sidebar.date_input(
        "选择时间范围",
        value=(default_start, default_end)
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

    # ============= 可视化：柱+折线 =============
    st.header("📊 问题类型对比图（柱=回复/时效，线=满意度）")

    level_choice = st.selectbox("选择问题层级", ["一级问题", "二级问题"], index=0)
    cur_df = lvl1 if level_choice == "一级问题" else lvl2

    cur_df = cur_df.dropna(subset=["回复次数_P90", "处理时长_P90", "满意度_4_5占比"])
    if not cur_df.empty:
        x_col = "class_one" if level_choice == "一级问题" else "class_two"
        metrics = ["回复次数_P90", "处理时长_P90", "满意度_4_5占比"]

        df_plot = cur_df.copy()
        for m in metrics:
            df_plot[m] = pd.to_numeric(df_plot[m], errors="coerce")
            df_plot[m + "_norm"] = (
                (df_plot[m] - df_plot[m].min()) / (df_plot[m].max() - df_plot[m].min())
                if df_plot[m].max() != df_plot[m].min() else df_plot[m]
            )

        # 仅对数值列求平均，避免对象列报错
        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
        df_plot = df_plot.groupby(x_col, as_index=False)[numeric_cols].mean()

        problem_choices = sorted(df_plot[x_col].unique())
        selected_problems = st.multiselect(
            f"选择要显示的{level_choice}（默认显示前15项）",
            problem_choices,
            default=problem_choices[:15]
        )
        if selected_problems:
            df_plot = df_plot[df_plot[x_col].isin(selected_problems)]

        bar_df = df_plot.melt(
            id_vars=[x_col],
            value_vars=["回复次数_P90_norm", "处理时长_P90_norm"],
            var_name="指标",
            value_name="标准化数值"
        ).replace({"回复次数_P90_norm": "回复次数P90", "处理时长_P90_norm": "处理时长P90"})

        fig = go.Figure()
        for metric, color in zip(["回复次数P90", "处理时长P90"], ["#5B8FF9", "#5AD8A6"]):
            data = bar_df[bar_df["指标"] == metric]
            fig.add_trace(go.Bar(
                x=data[x_col], y=data["标准化数值"], name=metric,
                marker_color=color,
                text=[f"{v:.2f}" for v in data["标准化数值"]],
                textposition="outside"
            ))

        fig.add_trace(go.Scatter(
            x=df_plot[x_col], y=df_plot["满意度_4_5占比_norm"],
            name="满意度(4/5占比)",
            mode="lines+markers+text",
            line=dict(color="#F6BD16", width=3),
            marker=dict(size=8),
            text=[f"{v:.2f}" for v in df_plot["满意度_4_5占比_norm"]],
            textposition="top center"
        ))

        fig.update_layout(
            title=f"{level_choice}：各问题类型 三指标对比（柱=回复/时效，线=满意度）",
            barmode="group",
            xaxis_title="问题类型",
            yaxis_title="标准化数值(0~1)",
            xaxis_tickangle=-30,
            plot_bgcolor="white",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============= 🏆 Top5 榜单 =============
    st.markdown("<h2 style='text-align:center; color:#2B3A67;'>🏆 Top5 榜单</h2>", unsafe_allow_html=True)

    # 自动根据层级选字段
    x_col = "class_one" if level_choice == "一级问题" else "class_two"

    # 先聚合一次，防止重复问题多条记录
    if not cur_df.empty:
        df_rank = (
            cur_df.groupby(x_col, as_index=False)
            .agg({
                "处理时长_P90": "mean",
                "满意度_4_5占比": "mean",
                "样本量": "sum"
            })
        )
    else:
        df_rank = pd.DataFrame(columns=[x_col, "处理时长_P90", "满意度_4_5占比", "样本量"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<h3 style='color:#2B3A67;'>⏱️ 处理时长最慢 Top5（按{x_col}）</h3>", unsafe_allow_html=True)
        if df_rank.empty:
            st.info("暂无数据")
        else:
            top5_slow = df_rank.sort_values("处理时长_P90", ascending=False).head(5)
            st.dataframe(
                top5_slow[[x_col, "处理时长_P90", "样本量"]]
                .rename(columns={x_col: "问题类型"})
                .reset_index(drop=True),
                use_container_width=True
            )

    with col2:
        st.markdown(f"<h3 style='color:#2B3A67;'>😞 满意度最低 Top5（按{x_col}）</h3>", unsafe_allow_html=True)
        if df_rank.empty:
            st.info("暂无数据")
        else:
            top5_bad = df_rank.sort_values("满意度_4_5占比", ascending=True).head(5)
            st.dataframe(
                top5_bad[[x_col, "满意度_4_5占比", "样本量"]]
                .rename(columns={x_col: "问题类型"})
                .reset_index(drop=True),
                use_container_width=True
            )

    # ============= 🌍 热力图分析 =============
    st.header("🌍 维度交叉热力图（满意度 or 时效）")

    if not df_f.empty:
        st.markdown("展示不同维度组合下的关键指标表现，可用于横向比较渠道、国家或业务线。")

        # 选择维度与指标
        x_dim = st.selectbox("选择 X 轴维度", ["business_line", "ticket_channel", "site_code"], index=0)
        y_dim = st.selectbox("选择 Y 轴维度", ["ticket_channel", "site_code", "business_line"], index=1)
        metric_sel = st.radio("选择指标", ["满意度_4_5占比", "处理时长_P90", "回复次数_P90"], horizontal=True)

        if x_dim == y_dim:
            st.warning("⚠️ X 轴与 Y 轴不能相同，请选择不同维度。")
        else:
            # 计算热力图数据
            df_hm = group_metrics(df_f.copy(), [], [x_dim, y_dim]).pivot(index=y_dim, columns=x_dim, values=metric_sel)

            if df_hm.empty:
                st.info("暂无数据可绘制热力图，请调整筛选条件。")
            else:
                # 保证轴为字符串，避免序列化问题
                x_vals = [str(v) for v in df_hm.columns.tolist()]
                y_vals = [str(v) for v in df_hm.index.tolist()]
                z_vals = df_hm.values

                # 准备单元格文本
                z_text = pd.DataFrame(z_vals, index=y_vals, columns=x_vals).round(2).astype(str).values

                fig_hm = go.Figure(
                    data=go.Heatmap(
                        z=z_vals,
                        x=x_vals,
                        y=y_vals,
                        colorscale="RdYlBu_r",
                        # ✅ 合法的 colorbar 配置（不使用 textfont）
                        colorbar=dict(
                            title=str(metric_sel),
                            titlefont=dict(size=16, color="black"),
                            tickfont=dict(size=14, color="black")
                        ),
                        hovertemplate=f"{x_dim}: %{{x}}<br>{y_dim}: %{{y}}<br>{metric_sel}: %{{z:.3f}}<extra></extra>",
                        text=z_text,
                        texttemplate="%{text}",
                        # textfont 在 Heatmap 级别是被支持的；如遇老版本兼容问题，可注释掉下一行
                        textfont=dict(size=14, color="black")
                    )
                )

                fig_hm.update_layout(
                    title=dict(
                        text=f"{metric_sel} - {x_dim} × {y_dim} 热力图",
                        font=dict(size=20, color="#2B3A67"),
                        x=0.5,
                        xanchor="center"
                    ),
                    xaxis=dict(
                        title=x_dim,
                        tickangle=-30,
                        tickfont=dict(size=14),
                        titlefont=dict(size=16)
                    ),
                    yaxis=dict(
                        title=y_dim,
                        tickfont=dict(size=14),
                        titlefont=dict(size=16)
                    ),
                    plot_bgcolor="white",
                    height=700,
                    margin=dict(l=80, r=80, t=80, b=80)
                )

                st.plotly_chart(fig_hm, use_container_width=True)

    # ============= 导出报告 =============
    st.header("📤 导出分析报告")
    filters_text = f"时间范围：{start_date} ~ {end_date}；业务线：{bl_sel or '全部'}；渠道：{ch_sel or '全部'}；国家：{site_sel or '全部'}"
    buffer = BytesIO()
    export_sheets(buffer, {"一级问题": lvl1, "二级问题": lvl2}, filters_text)
    st.download_button(
        "📥 下载带筛选说明的Excel报告",
        data=buffer,
        file_name="客服问题层级分析报告.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("请上传包含【评分(1-5)】【处理时长】【message_count】【site_code】的数据文件。")

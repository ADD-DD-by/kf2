# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import plotly.express as px

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
    st.header("问题类型对比图（柱=回复/时效，线=满意度）")
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
    # ============= 🔍 单问题明细散点：回复次数/处理时长 vs 评分 =============
    st.header("🔍 单问题分类：明细散点（回复次数/处理时长 vs 评分）")
    st.markdown("选择一个问题分类，查看每条样本在 **回复次数 或 处理时长** 与 **评分** 之间的关系。")

    # 选择层级 & 问题分类
    detail_level = st.radio("选择问题层级用于散点", ["一级问题", "二级问题"], horizontal=True)
    problem_field = "class_one" if detail_level == "一级问题" else "class_two"

    # 仅从筛选后的原始数据 df_f 中取（不是汇总表），保证是一条一条样本点
    if problem_field not in df_f.columns:
        st.info(f"当前数据没有字段：{problem_field}")
    else:
        # 允许用户选具体的一个问题分类
        problem_list = sorted(df_f[problem_field].dropna().unique().tolist())
        if not problem_list:
            st.info("当前筛选下没有可选的问题分类。")
        else:
            picked_problem = st.selectbox(f"选择{detail_level}", problem_list)

            # 选择横轴：回复次数 or 处理时长
            x_choice = st.radio("选择横轴指标", ["回复次数（message_count）", "处理时长（处理时长）"], horizontal=True)
            x_col_raw = "message_count" if "回复次数" in x_choice else "处理时长"

            # 取到该问题分类的样本明细
            pts = df_f[df_f[problem_field] == picked_problem].copy()

            # 需要的列：x_col_raw 与 评分
            need_cols = [x_col_raw, "评分"]
            pts = pts.dropna(subset=[c for c in need_cols if c in pts.columns])

            # 类型安全：再次转数值
            if x_col_raw in pts.columns:
                pts[x_col_raw] = pd.to_numeric(pts[x_col_raw], errors="coerce")
            if "评分" in pts.columns:
                pts["评分"] = pd.to_numeric(pts["评分"], errors="coerce")
            pts = pts.dropna(subset=need_cols)

            # 可选：轻微抖动，避免遮挡
            add_jitter = st.checkbox("为散点添加轻微抖动以减少遮挡", value=True)
            if add_jitter:
                rng = np.random.default_rng(42)
                # 仅对数值列加非常小的噪声
                pts["_x"] = pts[x_col_raw].astype(float) + rng.normal(0, max(pts[x_col_raw].std() * 0.01, 1e-6), len(pts))
                pts["_y"] = pts["评分"].astype(float) + rng.normal(0, 0.02, len(pts))
            else:
                pts["_x"] = pts[x_col_raw].astype(float)
                pts["_y"] = pts["评分"].astype(float)

            if pts.empty:
                st.info("该问题分类下没有足够的数据点。")
            else:
                # 计算相关系数与样本数
                try:
                    r = np.corrcoef(pts[x_col_raw], pts["评分"])[0, 1]
                except Exception:
                    r = np.nan

                st.markdown(f"📈 **相关系数 r = {r:.3f}**（{x_col_raw} 与 评分） | 样本数：**{len(pts)}**")

                # 趋势线（线性回归）
                trend_x = pts[x_col_raw].to_numpy(dtype=float)
                trend_y = pts["评分"].to_numpy(dtype=float)
                line_trace = None
                if len(pts) > 2 and np.isfinite(trend_x).all() and np.isfinite(trend_y).all():
                    z = np.polyfit(trend_x, trend_y, 1)
                    p = np.poly1d(z)
                    # 为了趋势线更平滑，按范围画
                    xs = np.linspace(trend_x.min(), trend_x.max(), 100)
                    ys = p(xs)
                else:
                    xs, ys = None, None

                # 颜色按渠道/国家/业务线三选一（可选）
                color_dim = st.selectbox("散点着色维度（可选）", ["不着色", "渠道 ticket_channel", "国家 site_code", "业务线 business_line"], index=0)
                if color_dim == "不着色":
                    color_vals = None
                    legend_name = None
                else:
                    dim_map = {
                        "渠道 ticket_channel": "ticket_channel",
                        "国家 site_code": "site_code",
                        "业务线 business_line": "business_line",
                    }
                    legend_name = dim_map[color_dim]
                    if legend_name in pts.columns:
                        color_vals = pts[legend_name].fillna("未知").astype(str)
                    else:
                        color_vals = None
                        legend_name = None

                fig_det = go.Figure()

                if color_vals is None:
                    # 单色散点
                    fig_det.add_trace(go.Scattergl(
                        x=pts["_x"], y=pts["_y"],
                        mode="markers",
                        name=picked_problem,
                        marker=dict(size=9, color="#5B8FF9", opacity=0.65, line=dict(width=0.5, color="gray")),
                        hovertemplate=f"{detail_level}: {picked_problem}<br>{x_col_raw}: %{{x:.2f}}<br>评分: %{{y:.2f}}<extra></extra>"
                    ))
                else:
                    # 分组着色：每个类别一条 trace
                    for val in sorted(color_vals.unique()):
                        sub = pts[color_vals == val]
                        fig_det.add_trace(go.Scattergl(
                            x=sub["_x"], y=sub["_y"],
                            mode="markers",
                            name=str(val),
                            marker=dict(size=9, opacity=0.65, line=dict(width=0.5, color="gray")),
                            hovertemplate=f"{legend_name}: {val}<br>{x_col_raw}: %{{x:.2f}}<br>评分: %{{y:.2f}}<extra></extra>"
                        ))

                # 加趋势线
                if xs is not None:
                    fig_det.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines",
                        name="趋势线", line=dict(color="gray", width=2, dash="dot")
                    ))

                # 中位数参考线（可选）
                show_ref = st.checkbox("显示中位数参考线", value=False)
                if show_ref:
                    fig_det.add_hline(y=float(np.median(trend_y)), line=dict(color="#999999", width=1, dash="dash"), annotation_text="评分中位数")
                    fig_det.add_vline(x=float(np.median(trend_x)), line=dict(color="#999999", width=1, dash="dash"), annotation_text=f"{x_col_raw}中位数")

                fig_det.update_layout(
                    title=f"{detail_level}：{picked_problem} —— {x_col_raw} vs 评分（明细散点）",
                    xaxis_title=x_col_raw,
                    yaxis_title="评分（1~5）",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=640,
                    title_x=0.5,
                    title_font=dict(size=20, color="#2B3A67"),
                    legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                )
                # 限制评分轴范围（可读性更好）
                fig_det.update_yaxes(range=[0.5, 5.5])
                st.plotly_chart(fig_det, use_container_width=True)
    # ============= 各问题相关性分析（找出正/负相关最强的问题） =============
    st.header("各问题分类相关性分析（回复次数/处理时长 vs 评分）")
    st.markdown("自动计算所有问题分类中【回复次数/处理时长】与【评分】的相关系数，找出正/负相关最强的问题。")

    # 选择层级
    corr_level = st.radio("选择层级", ["一级问题", "二级问题"], horizontal=True, key="corr_level_radio")
    problem_field = "class_one" if corr_level == "一级问题" else "class_two"

    # 数据准备：从 df_f 中取明细
    if problem_field not in df_f.columns:
        st.info(f"当前数据中未找到字段：{problem_field}")
    else:
        df_corr = df_f.copy().dropna(subset=[problem_field, "评分"])
        df_corr["评分"] = pd.to_numeric(df_corr["评分"], errors="coerce")
        df_corr["处理时长"] = pd.to_numeric(df_corr.get("处理时长", np.nan), errors="coerce")
        df_corr["message_count"] = pd.to_numeric(df_corr.get("message_count", np.nan), errors="coerce")

        # 选择分析的指标
        metric_sel = st.selectbox("选择用于计算相关系数的指标", ["回复次数（message_count）", "处理时长（处理时长）"], index=0)
        metric_col = "message_count" if "回复次数" in metric_sel else "处理时长"

        # 计算每个问题的相关系数
        corr_list = []
        for pb, sub in df_corr.groupby(problem_field):
            sub = sub.dropna(subset=[metric_col, "评分"])
            if len(sub) >= 5:  # 至少5条样本再算
                try:
                    r = np.corrcoef(sub[metric_col], sub["评分"])[0, 1]
                    corr_list.append((pb, len(sub), r))
                except Exception:
                    pass

        if not corr_list:
            st.warning("暂无足够数据计算相关系数。")
        else:
            df_r = pd.DataFrame(corr_list, columns=["问题分类", "样本量", "相关系数"])
            df_r["相关系数"] = df_r["相关系数"].round(3)
            df_r = df_r.sort_values("相关系数", ascending=False).reset_index(drop=True)

            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 正相关最高 Top5（评分随指标升高而升高）")
                st.dataframe(df_r.head(5), use_container_width=True)
            with col2:
                st.subheader("📉 负相关最高 Top5（评分随指标升高而下降）")
                st.dataframe(df_r.tail(5).iloc[::-1], use_container_width=True)

            # 绘制条形图
            show_bar = st.checkbox("显示所有问题的相关系数条形图", value=False)
            if show_bar:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=df_r["问题分类"],
                    y=df_r["相关系数"],
                    marker_color=np.where(df_r["相关系数"] > 0, "#5B8FF9", "#E8684A"),
                    text=df_r["相关系数"],
                    textposition="outside"
                ))
                fig_bar.update_layout(
                    title=f"{corr_level}：{metric_sel} 与 评分 的相关系数分布",
                    xaxis_title="问题分类",
                    yaxis_title="相关系数 r",
                    xaxis_tickangle=-30,
                    plot_bgcolor="white",
                    height=600,
                    title_x=0.5,
                    title_font=dict(size=20, color="#2B3A67"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    # ============= 💬 气泡图（按问题颜色区分，无大小） =============
    st.header("💬 指标与满意度关系（气泡图）")

    if not lvl1.empty or not lvl2.empty:
        st.markdown("展示不同问题下，回复次数或处理时长与满意度的关系（颜色区分问题类别，去除气泡大小差异）。")

        # 可选层级与指标
        bubble_level = st.radio("选择展示层级", ["一级问题", "二级问题"], horizontal=True)
        x_metric = st.selectbox("选择横轴指标", ["处理时长_P90", "回复次数_P90"], index=1)
        y_metric = "满意度_4_5占比"

        # 根据层级选择字段
        problem_field = "class_one" if bubble_level == "一级问题" else "class_two"

        # ✅ 保证使用对应层级的聚合结果
        cur_src = lvl1 if bubble_level == "一级问题" else lvl2
        df_bubble = cur_src.copy().dropna(subset=[x_metric, y_metric])

        if problem_field not in df_bubble.columns:
            st.warning(f"⚠️ 当前层级 {bubble_level} 的数据中未找到字段 {problem_field}")
        elif df_bubble.empty:
            st.warning("⚠️ 当前筛选条件下暂无可用数据。")
        else:
            # 聚合
            df_bubble = df_bubble.groupby(problem_field, as_index=False).agg({
                "处理时长_P90": "mean",
                "回复次数_P90": "mean",
                "满意度_4_5占比": "mean"
            })

            fig_bubble = go.Figure()

            # ✅ 安全颜色映射
            categories = sorted(df_bubble[problem_field].dropna().unique())
            palette = (px.colors.qualitative.Set3 if hasattr(px.colors.qualitative, "Set3")
                       else px.colors.qualitative.Set2)
            palette = palette * (len(categories) // len(palette) + 1)
            color_map = {cat: palette[i] for i, cat in enumerate(categories)}

            # 绘制
            for pb in categories:
                data = df_bubble[df_bubble[problem_field] == pb]
                fig_bubble.add_trace(go.Scatter(
                    x=data[x_metric],
                    y=data[y_metric],
                    mode="markers+text",
                    name=str(pb),
                    text=[pb],
                    textposition="top center",
                    marker=dict(
                        size=16,
                        color=color_map[pb],
                        line=dict(width=1, color="gray"),
                        opacity=0.9
                    ),
                    hovertemplate=(
                        f"{problem_field}: %{{text}}<br>"
                        f"{x_metric}: %{{x:.2f}}<br>"
                        f"{y_metric}: %{{y:.2f}}<extra></extra>"
                    )
                ))

            # 趋势线
            if len(df_bubble) > 2:
                z = np.polyfit(df_bubble[x_metric], df_bubble[y_metric], 1)
                p = np.poly1d(z)
                fig_bubble.add_trace(go.Scatter(
                    x=df_bubble[x_metric],
                    y=p(df_bubble[x_metric]),
                    mode="lines",
                    line=dict(color="gray", dash="dot"),
                    name="趋势线"
                ))

            # 相关系数
            if df_bubble[x_metric].nunique() > 1 and df_bubble[y_metric].nunique() > 1:
                corr = df_bubble[[x_metric, y_metric]].corr().iloc[0, 1]
                st.markdown(f"📈 **相关系数 r = {corr:.3f}** （{x_metric} 与 {y_metric}）")
            else:
                st.markdown("⚠️ 样本不足或数据无差异，无法计算相关系数。")

            fig_bubble.update_layout(
                title=f"{bubble_level}：{x_metric} 与 {y_metric} 的关系（按问题颜色区分）",
                xaxis_title=x_metric,
                yaxis_title=y_metric,
                plot_bgcolor="white",
                height=650,
                title_x=0.5,
                title_font=dict(size=20, color="#2B3A67"),
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

       # ============= 📈 月度趋势图（可选指标、层级、筛选） =============
    st.header("📈 指标月度趋势分析")

    if "month" in df_f.columns:
        st.markdown("用于分析不同问题在时间维度上的表现趋势，可选择指标、层级和筛选维度。")

        # 用户选择层级、指标、维度
        trend_level = st.radio("选择层级", ["一级问题", "二级问题"], horizontal=True)
        trend_metric = st.selectbox("选择趋势指标", ["处理时长_P90", "回复次数_P90", "满意度_4_5占比"], index=0)
        trend_dim = st.selectbox("选择分组维度", ["问题分类", "业务线", "渠道", "国家"], index=0)

        # 对应字段映射
        problem_field = "class_one" if trend_level == "一级问题" else "class_two"
        df_trend = lvl1 if trend_level == "一级问题" else lvl2

        if df_trend.empty:
            st.info("暂无数据")
        else:
            # 确定分组字段
            if trend_dim == "问题分类":
                group_field = problem_field
            elif trend_dim == "业务线":
                group_field = "business_line"
            elif trend_dim == "渠道":
                group_field = "ticket_channel"
            else:
                group_field = "site_code"

            # 清理并聚合：避免同月多维重复
            use_cols = [c for c in ["month", group_field, trend_metric] if c in df_trend.columns]
            df_trend = df_trend[use_cols].dropna(subset=[trend_metric])
            df_trend = df_trend.groupby(["month", group_field], as_index=False).mean()

            # 选择显示的前若干类
            top_groups = sorted(df_trend[group_field].unique())
            sel_groups = st.multiselect(f"选择要显示的{trend_dim}",
                                        top_groups,
                                        default=top_groups[:5])
            df_trend = df_trend[df_trend[group_field].isin(sel_groups)]

            if df_trend.empty:
                st.warning("当前筛选条件下无数据。")
            else:
                fig_trend = go.Figure()
                for gp in sel_groups:
                    data = df_trend[df_trend[group_field] == gp]
                    fig_trend.add_trace(go.Scatter(
                        x=data["month"],
                        y=data[trend_metric],
                        mode="lines+markers+text",
                        name=str(gp),
                        text=[f"{v:.2f}" for v in data[trend_metric]],
                        textposition="top center",
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))

                fig_trend.update_layout(
                    title=f"{trend_level}：{trend_metric} 按 {trend_dim} 的月度趋势",
                    xaxis_title="月份",
                    yaxis_title=trend_metric,
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

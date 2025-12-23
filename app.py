import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
import textwrap

# --- 基础库检查 ---
try:
    import yfinance as yf
    import numpy as np
except ImportError as e:
    st.error(f"缺少必要库，请先安装: {e}")
    st.stop()

# --- 页面设置 ---
st.set_page_config(page_title="股价复盘 (最终修复版)", layout="wide")
st.title("📈 2025 股价复盘系统：最终修复版")
st.markdown("---")

# --- 0. 代理设置 (修改：默认为关闭，适应云端环境) ---
st.sidebar.header("0. 网络代理设置")
# 【关键修改】默认 value 改为 False，防止云端部署时报错
enable_proxy = st.sidebar.checkbox("开启代理连接 (本地运行需勾选)", value=False)
proxy_address = st.sidebar.text_input("代理地址", value="http://127.0.0.1:17890")

if enable_proxy:
    os.environ["HTTP_PROXY"] = proxy_address
    os.environ["HTTPS_PROXY"] = proxy_address
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
else:
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

# --- 1. 数据来源 ---
st.sidebar.header("1. 数据来源")
data_source = st.sidebar.radio("选择模式", ["Yahoo Finance (实盘数据)", "Excel文件 (Prices表)", "生成模拟数据 (测试用)"])

# --- 2. 绘图参数 ---
st.sidebar.header("2. 绘图参数")
default_start = pd.to_datetime("2024-12-23")
default_end = min(pd.to_datetime("2025-12-23"), datetime.today())

ticker = st.sidebar.text_input("股票代码", value="6324.T")
start_date = st.sidebar.date_input("开始日期", value=default_start)
end_date_input = st.sidebar.date_input("结束日期", value=default_end, max_value=datetime.today())
end_date_final = end_date_input + timedelta(days=1)

# --- 3. 视觉与排版微调 ---
st.sidebar.header("3. 视觉与排版微调")
st.sidebar.info("💡 提示：如果下载后字体太小，请调大【导出倍率】下方的字体滑块，或将倍率设为 1x。")

st.sidebar.subheader("🖨️ 导出设置 (关键)")
export_scale = st.sidebar.radio(
    "导出清晰度/倍率", 
    [1, 2, 3], 
    index=0, 
    format_func=lambda x: f"{x}倍 (1倍=所见即所得, 3倍=字会变小但超清)",
    horizontal=True
)

phase_font_size = st.sidebar.slider("顶部阶段字体大小", 10, 80, 20)
event_font_size = st.sidebar.slider("下方事件字体大小", 8, 60, 16)

phase_label_y = st.sidebar.slider("阶段标签基础高度", 1.0, 1.3, 1.02, 0.01)
phase_stagger = st.sidebar.checkbox("开启顶部标签错落", value=True)
phase_stagger_gap = st.sidebar.slider("顶部错落高度差", 0.01, 0.15, 0.05)

label_wrap_width = st.sidebar.slider("标签换行字数", 5, 30, 10)
hover_wrap_width = st.sidebar.slider("悬浮文字换行字数", 20, 80, 40)

arrow_len_base = st.sidebar.slider("引线基础长度", 20, 150, 50)
stagger_steps = st.sidebar.slider("下方防重叠阶梯数", 3, 10, 6)
stagger_gap = st.sidebar.slider("下方阶梯垂直间距", 10, 100, 50)

y_headroom = st.sidebar.slider("顶部强制留白 (%)", 0, 100, 7)
bg_opacity = st.sidebar.slider("标签背景透明度", 0.1, 1.0, 0.8)
bottom_margin = st.sidebar.slider("底部留白高度", 50, 150, 80)
top_margin = st.sidebar.slider("顶部留白高度", 100, 300, 150)

# --- 4. 上传文件 ---
st.sidebar.header("4. 上传文件")
uploaded_file = st.sidebar.file_uploader("上传 Excel (中文版)", type=["xlsx"])

# --- 辅助函数 ---
def process_text_smart(text, wrap_width):
    if not isinstance(text, str): return str(text)
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        line = line.replace("<br>", "\n")
        sub_lines = line.split("\n")
        for sl in sub_lines:
            wrapped = textwrap.wrap(sl, width=wrap_width)
            processed_lines.extend(wrapped)
    return "<br>".join(processed_lines)

def generate_mock_data(start, end):
    dates = pd.date_range(start=start, end=end, freq='B')
    n = len(dates)
    if n == 0: return None
    np.random.seed(42)
    returns = np.random.normal(loc=0.0003, scale=0.015, size=n)
    price = 3000 * np.cumprod(1 + returns)
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = df['Close'].shift(1).fillna(price[0]) * (1 + np.random.randn(n)*0.005)
    return df.round(0)

def load_data_from_excel(file):
    try:
        df = pd.read_excel(file, sheet_name='Prices')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except: 
        return None

def get_stock_data(source, ticker, start, end, uploaded_file):
    if source == "Yahoo Finance (实盘数据)":
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        try:
            with st.spinner("正在连接 Yahoo..."):
                dat = yf.Ticker(ticker)
                df = dat.history(start=start_str, end=end_str, auto_adjust=True)
            if df.empty:
                st.error("❌ Yahoo 返回空数据")
                return None
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            st.error(f"连接失败: {e}")
            return None
    elif source == "Excel文件 (Prices表)":
        return load_data_from_excel(uploaded_file) if uploaded_file else None
    else:
        return generate_mock_data(start, end)

# --- 智能解析与聚合函数 ---
def find_col_in_list(columns, keywords, exclude_keywords=None):
    for col in columns:
        col_str = str(col)
        if exclude_keywords and any(ex in col_str for ex in exclude_keywords):
            continue
        for kw in keywords:
            if kw in col_str:
                return col
    return None

def extract_table_dynamically(df, required_keywords, name="Table"):
    def check_columns(cols):
        found_cols = {}
        for key, (kws, ex_kws) in required_keywords.items():
            found = find_col_in_list(cols, kws, ex_kws)
            if found:
                found_cols[key] = found
            else:
                return None
        return found_cols

    found_cols = check_columns(df.columns)
    if found_cols: return df, found_cols

    max_scan = min(len(df), 100)
    for i in range(max_scan):
        row_values = df.iloc[i].astype(str).tolist()
        is_header_row = True
        for key, (kws, ex_kws) in required_keywords.items():
            if not any(kw in cell for cell in row_values for kw in kws):
                is_header_row = False
                break
        
        if is_header_row:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i]
            new_found_cols = check_columns(new_df.columns)
            if new_found_cols:
                return new_df, new_found_cols
    return None, None

def aggregate_details(df, group_keys, detail_col, output_detail_name="Detail"):
    if not detail_col: return df
    for k in group_keys:
        df[k] = df[k].ffill()
    
    def join_text(series):
        texts = [str(s).strip() for s in series if pd.notna(s) and str(s).strip() != '']
        if not texts: return None
        if len(texts) == 1: return texts[0]
        return "<br>".join([f"• {t}" for t in texts])

    agg_dict = {detail_col: join_text}
    temp = df.groupby(group_keys, as_index=False).agg(agg_dict)
    temp = temp.rename(columns={detail_col: output_detail_name})
    return temp

def parse_uploaded_excel(file):
    try:
        all_sheets = pd.read_excel(file, sheet_name=None)
        events_list = []
        phases_list = []
        
        event_rules = {
            'event': (['主要驱动', 'Event'], None),
            'date': (['日期', 'Date', '时间'], ['起始', '开始', 'Start', '结束', 'End'])
        }
        phase_rules = {
            'phase': (['阶段概述', 'Phase'], None),
            'start': (['起始日期', '开始日期', 'Start'], None),
            'end': (['结束日期', 'End'], None)
        }

        for sheet_name, df in all_sheets.items():
            df.columns = df.columns.astype(str).str.strip()
            
            # 1. 提取事件
            e_df, e_cols = extract_table_dynamically(df, event_rules, "Events")
            if e_df is not None:
                hover_col = find_col_in_list(e_df.columns, ['详细解释', '因果链', 'Detailed'])
                cols_to_keep = [e_cols['date'], e_cols['event']]
                if hover_col: cols_to_keep.append(hover_col)
                temp = e_df[cols_to_keep].copy()
                
                if hover_col:
                    temp = aggregate_details(temp, [e_cols['date'], e_cols['event']], hover_col, '详细解释')
                    temp = temp.rename(columns={e_cols['date']: 'Date', e_cols['event']: '主要驱动'})
                else:
                    temp = temp.rename(columns={e_cols['date']: 'Date', e_cols['event']: '主要驱动'})

                # 【核心修复】：使用 errors='coerce' 避免 "起始日期" 文本报错
                temp['Date'] = pd.to_datetime(temp['Date'], errors='coerce')
                temp = temp.dropna(subset=['Date'])
                if not temp.empty: events_list.append(temp)
            
            # 2. 提取阶段
            p_df, p_cols = extract_table_dynamically(df, phase_rules, "Phases")
            if p_df is not None:
                hover_col = find_col_in_list(p_df.columns, ['关键因素', '要点', 'Key Factors'])
                cols_to_keep = [p_cols['start'], p_cols['end'], p_cols['phase']]
                if hover_col: cols_to_keep.append(hover_col)
                temp = p_df[cols_to_keep].copy()
                
                if hover_col:
                    temp = aggregate_details(temp, [p_cols['start'], p_cols['end'], p_cols['phase']], hover_col, '关键因素')
                    temp = temp.rename(columns={p_cols['start']: 'Start date', p_cols['end']: 'End date', p_cols['phase']: '阶段概述'})
                else:
                    temp = temp.rename(columns={p_cols['start']: 'Start date', p_cols['end']: 'End date', p_cols['phase']: '阶段概述'})
                
                # 【核心修复】：使用 errors='coerce'
                temp['Start date'] = pd.to_datetime(temp['Start date'], errors='coerce')
                temp['End date'] = pd.to_datetime(temp['End date'], errors='coerce')
                temp = temp.dropna(subset=['Start date'])
                if not temp.empty: phases_list.append(temp)

        events_df = pd.concat(events_list, ignore_index=True) if events_list else None
        phases_df = pd.concat(phases_list, ignore_index=True) if phases_list else None
        return events_df, phases_df

    except Exception as e:
        import traceback
        st.error(f"解析 Excel 出错: {e}")
        st.text(traceback.format_exc())
        return None, None

# --- 主程序 ---
if uploaded_file:
    stock_df = get_stock_data(data_source, ticker, start_date, end_date_final, uploaded_file)
    
    if stock_df is not None and not stock_df.empty:
        events_df, phases_df = parse_uploaded_excel(uploaded_file)
        
        if events_df is None and phases_df is None:
            st.warning("⚠️ 未能识别内容。请确保Excel包含：'主要驱动'或'阶段概述'列。")
        else:
            try:
                fig = go.Figure()

                # 绘制股价
                fig.add_trace(go.Scatter(
                    x=stock_df.index, y=stock_df['Close'],
                    mode='lines', name=f"{ticker} 收盘价",
                    line=dict(color='#1976D2', width=2.5), line_shape='spline'
                ))
                data_start, data_end = stock_df.index.min(), stock_df.index.max()

                # 绘制阶段
                if phases_df is not None and not phases_df.empty:
                    phase_colors = ["rgba(255,99,132,0.12)", "rgba(54,162,235,0.12)", "rgba(255,206,86,0.15)", "rgba(75,192,192,0.12)"]
                    target_col = find_col_in_list(phases_df.columns, ['阶段概述'])
                    for i, row in phases_df.iterrows():
                        p_start = max(row['Start date'], data_start)
                        p_end = min(row['End date'], data_end)
                        if p_start < p_end:
                            mid_point = p_start + (p_end - p_start) / 2
                            fig.add_vrect(x0=p_start, x1=p_end, fillcolor=phase_colors[i % 4], layer="below", line_width=0)
                            
                            raw_text = str(row.get(target_col, ''))
                            wrapped_text = process_text_smart(raw_text, label_wrap_width)
                            
                            hover_col = find_col_in_list(phases_df.columns, ['关键因素', '要点', 'Key Factors'])
                            hover_text_raw = str(row.get(hover_col, '')) if hover_col else raw_text
                            hover_text = process_text_smart(hover_text_raw, hover_wrap_width)
                            
                            current_phase_y = phase_label_y
                            if phase_stagger: current_phase_y += (i % 2) * phase_stagger_gap

                            fig.add_annotation(
                                x=mid_point, y=current_phase_y, yref="paper", 
                                text=f"<b>{wrapped_text}</b>", hovertext=hover_text,
                                showarrow=False, font=dict(size=phase_font_size, color="#555"), 
                                bgcolor="rgba(255,255,255,0.8)", borderpad=3
                            )

                # 绘制事件
                if events_df is not None and not events_df.empty:
                    events_df = events_df.sort_values('Date').reset_index(drop=True)
                    label_col = find_col_in_list(events_df.columns, ['主要驱动'])
                    for i, row in events_df.iterrows():
                        event_date = row['Date']
                        if data_start <= event_date <= data_end:
                            try:
                                idx = stock_df.index.get_indexer([event_date], method='nearest')[0]
                                curr = stock_df.index[idx]
                                vals = stock_df.loc[curr]
                                close_p = vals['Close'].iloc[0] if isinstance(vals['Close'], pd.Series) else vals['Close']
                                open_p = vals['Open'].iloc[0] if isinstance(vals['Open'], pd.Series) else vals['Open']
                                
                                y_anchor = close_p
                                is_rising = close_p >= open_p
                                ay_dir = 1 if is_rising else -1
                                color = "#D32F2F" if is_rising else "#00796B"
                                stagger_level = i % stagger_steps 
                                current_arrow_len = arrow_len_base + (stagger_level * stagger_gap)
                                
                                txt = str(row.get(label_col, ''))
                                formatted = process_text_smart(txt, label_wrap_width)
                                hover_col = find_col_in_list(events_df.columns, ['详细解释', '因果链', 'Detailed'])
                                hover_text_raw = str(row.get(hover_col, '')) if hover_col else txt
                                hover_formatted = process_text_smart(hover_text_raw, hover_wrap_width)
                                
                                fig.add_annotation(
                                    x=curr, y=y_anchor, text=f"<b>{formatted}</b>",
                                    hovertext=hover_formatted, 
                                    showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=color,
                                    ax=0, ay=current_arrow_len * ay_dir,
                                    font=dict(size=event_font_size, color="#333"), 
                                    bgcolor=f"rgba(255,255,255,{bg_opacity})", 
                                    bordercolor=color, borderwidth=1, borderpad=3,
                                    hoverlabel=dict(bgcolor="white", font=dict(size=event_font_size))
                                )
                            except: pass

                # 布局
                y_max = stock_df['Close'].max()
                y_min = stock_df['Close'].min()
                range_max = y_max * (1 + y_headroom / 100)
                range_min = y_min * 0.95

                fig.update_layout(
                    title=dict(text=f"{ticker} 收盘价趋势复盘", x=0.5, font=dict(size=22)),
                    yaxis_title="收盘价 (JPY)",
                    height=950, xaxis_rangeslider_visible=False,
                    template="plotly_white", margin=dict(t=top_margin, r=50, b=bottom_margin), 
                    plot_bgcolor='rgba(250,250,250,1)', hovermode="x unified", dragmode="pan"
                )
                fig.update_xaxes(tickformat="%y年%-m月", dtick="M1", showgrid=True, gridcolor='rgba(0,0,0,0.05)')
                fig.update_yaxes(range=[range_min, range_max], showgrid=True, gridcolor='rgba(0,0,0,0.05)')

                st.plotly_chart(fig, use_container_width=True, config={
                    'editable': True, 'scrollZoom': True,
                    'toImageButtonOptions': {
                        'format': 'png', 'filename': f'{ticker}_复盘分析',
                        'height': 950 * export_scale, 'width': 1600 * export_scale, 'scale': 1 
                    }
                })

            except Exception as e:
                import traceback
                st.error(f"绘图报错: {e}")
                st.text(traceback.format_exc())
    else:
        if data_source != "Yahoo Finance (实盘数据)" and (stock_df is None or stock_df.empty):
             st.warning("⚠️ 数据为空")
else:
    st.info("👈 请上传 Excel 文件")

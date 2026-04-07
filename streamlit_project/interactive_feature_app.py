import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ["WenQuanYi"," Micro Hei", "Noto Sans CJK"]
plt.rcParams['axes.unicode_minus'] = False
# 设置页面配置
st.set_page_config(page_title="特征预测分析", layout="wide")

# 自定义CSS样式
st.markdown("""
<style>
    .feature-group {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
    }
    .feature-item {
        padding: 8px 0;
        font-size: 16px;
        color: #333;
    }
    .underline {
        border-bottom: 2px solid #FF6B6B;
        padding-bottom: 15px;
        margin-bottom: 15px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .safe-status {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-status {
        color: #FF9800;
        font-weight: bold;
    }
    .danger-status {
        color: #f44336;
        font-weight: bold;
    }
    .feature-toggle {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.title("🚄 轨道交通特征预测分析系统")
st.markdown("---")

# 定义特征数据（初始值）
initial_features_data = {
    "事故相关特征": [
        {"id": "feature₁", "name": "no accident occurs", "value": 1, "status": "safe"},
        {"id": "feature₂", "name": "the total duration of all accidents is 0", "value": 1, "status": "safe", "underline": True},
        {"id": "feature₃", "name": "the total duration of all accidents is 0 or relatively short", "value": 1, "status": "safe", "underline": True}
    ],
    "交通密度特征": [
        {"id": "feature₄", "name": "traffic density is in the lowest quartile", "value": 1, "status": "safe"},
        {"id": "feature₅", "name": "traffic density is in the lower half of the distribution", "value": 1, "status": "safe"},
        {"id": "feature₆", "name": "traffic density is below the highest quartile", "value": 1, "status": "warning", "underline": True}
    ],
    "列车冲突特征": [
        {"id": "feature₇", "name": "no conflicting train itinerary occurs", "value": 1, "status": "safe"},
        {"id": "feature₈", "name": "the level of conflicting train itineraries is 0 or relatively low", "value": 1, "status": "safe", "underline": True}
    ],
    "非预期偏差特征": [
        {"id": "feature₉", "name": "no unanticipated deviation occurs", "value": 1, "status": "safe"},
        {"id": "feature₁₀", "name": "the level of unanticipated deviation is 0 or relatively low", "value": 1, "status": "safe", "underline": True}
    ],
    "工作负荷特征": [
        {"id": "feature₁₁", "name": "the expected workload of the current workstation during the current hour is in the lowest quintile", "value": 1, "status": "safe"},
        {"id": "feature₁₂", "name": "the expected workload of the current workstation during the current hour is in the lowest two quintiles", "value": 1, "status": "safe"},
        {"id": "feature₁₃", "name": "the expected workload of the current workstation during the current hour is in the lowest three quintiles", "value": 1, "status": "safe"},
        {"id": "feature₁₄", "name": "the expected workload of the current workstation during the current hour is below the highest quintile", "value": 1, "status": "warning", "underline": True}
    ],
    "时段特征": [
        {"id": "feature₁₅", "name": "this is a peak period", "value": 1, "status": "danger"}
    ]
}

# 侧边栏控制
st.sidebar.header("⚙️ 控制面板")
show_underline = st.sidebar.checkbox("显示分组下划线", value=True)
show_metrics = st.sidebar.checkbox("显示统计指标", value=True)
chart_type = st.sidebar.selectbox("图表类型", ["雷达图", "条形图", "热力图"])

# 添加重置按钮
if st.sidebar.button("🔄 重置所有特征为1"):
    for group_name in st.session_state.features_data:
        for feature in st.session_state.features_data[group_name]:
            st.session_state[f"toggle_{feature['id']}"] = True
    st.rerun()

if st.sidebar.button("⚠️ 重置所有特征为0"):
    for group_name in st.session_state.features_data:
        for feature in st.session_state.features_data[group_name]:
            st.session_state[f"toggle_{feature['id']}"] = False
    st.rerun()

# 初始化session_state
if 'features_data' not in st.session_state:
    st.session_state.features_data = initial_features_data

# 主内容区域
st.header("📊 特征预测结果展示")

# 创建两列布局
col1, col2 = st.columns([2, 1])

with col1:
    # 展示特征分组
    for group_name, features in st.session_state.features_data.items():
        with st.container():
            st.subheader(f"🔹 {group_name}")
            
            for i, feature in enumerate(features):
                # 创建两列：左边显示特征，右边放切换按钮
                feat_col, toggle_col = st.columns([4, 1])
                
                with toggle_col:
                    # 每个特征的可选切换
                    toggle_key = f"toggle_{feature['id']}"
                    if toggle_key not in st.session_state:
                        st.session_state[toggle_key] = feature['value'] == 1
                    
                    new_value = st.toggle(
                        "启用", 
                        value=st.session_state[toggle_key], 
                        key=toggle_key,
                        help=f"切换 {feature['id']} 的状态"
                    )
                    # 更新特征值
                    feature['value'] = 1 if new_value else 0
                
                with feat_col:
                    # 确定样式
                    underline_class = "underline" if (feature.get("underline") and show_underline) else ""
                    
                    # 状态图标（根据当前值显示）
                    if feature['value'] == 1:
                        if feature["status"] == "safe":
                            icon = "✅"
                            status_class = "safe-status"
                        elif feature["status"] == "warning":
                            icon = "⚠️"
                            status_class = "warning-status"
                        else:
                            icon = "🔴"
                            status_class = "danger-status"
                    else:
                        icon = "⭕"
                        status_class = ""
                    
                    # 显示特征
                    st.markdown(f"""
                    <div class="feature-item {underline_class}">
                        <b>{feature['id']} = {feature['value']}</b>: {feature['name']} 
                        <span class="{status_class}">{icon}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")

# 计算统计数据（移到外部，确保始终可用）
total_features = sum(len(features) for features in st.session_state.features_data.values())
active_count = sum(1 for features in st.session_state.features_data.values() for f in features if f['value'] == 1)

# 计算安全评分
safe_active = sum(1 for features in st.session_state.features_data.values() 
                 for f in features if f['value'] == 1 and f['status'] == 'safe')
warning_active = sum(1 for features in st.session_state.features_data.values() 
                    for f in features if f['value'] == 1 and f['status'] == 'warning')
danger_active = sum(1 for features in st.session_state.features_data.values() 
                   for f in features if f['value'] == 1 and f['status'] == 'danger')

safety_score = (safe_active / active_count * 100) if active_count > 0 else 0

with col2:
    if show_metrics:
        st.subheader("📈 实时统计概览")
        
        # 显示指标卡片
        st.markdown(f"""
        <div class="metric-card">
            <h3>激活特征数</h3>
            <h1>{active_count}/{total_features}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        col_safe, col_warn, col_danger = st.columns(3)
        with col_safe:
            st.metric(label="✅ 安全", value=safe_active)
        with col_warn:
            st.metric(label="⚠️ 警告", value=warning_active)
        with col_danger:
            st.metric(label="🔴 危险", value=danger_active)
        
        # 安全评分进度条
        st.progress(safety_score / 100)
        st.caption(f"当前安全评分: {safety_score:.1f}%")

# 可视化图表区域
st.markdown("---")
st.header("📉 特征可视化分析")

# 准备数据用于图表（基于当前值）
chart_data = []
for group_name, features in st.session_state.features_data.items():
    for feature in features:
        chart_data.append({
            'Group': group_name,
            'Feature': feature['id'],
            'Value': feature['value'],
            'Status': feature['status'],
            'Name': feature['name'][:30] + "..." if len(feature['name']) > 30 else feature['name'],
            'Active': feature['value'] == 1
        })

df = pd.DataFrame(chart_data)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

if chart_type == "条形图":
    colors = {'safe': '#4CAF50', 'warning': '#FF9800', 'danger': '#f44336', 'inactive': '#CCCCCC'}
    bar_colors = []
    for _, row in df.iterrows():
        if row['Active']:
            bar_colors.append(colors[row['Status']])
        else:
            bar_colors.append(colors['inactive'])
    
    ax.barh(df['Feature'], df['Value'], color=bar_colors)
    ax.set_xlabel('特征值 (0=关闭, 1=开启)')
    ax.set_title('特征状态分布（灰色=关闭）')
    ax.set_xlim(0, 1.5)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['safe'], label='安全(开启)'),
        Patch(facecolor=colors['warning'], label='警告(开启)'),
        Patch(facecolor=colors['danger'], label='危险(开启)'),
        Patch(facecolor=colors['inactive'], label='关闭')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

elif chart_type == "热力图":
    # 创建透视表
    pivot_data = df.pivot_table(values='Value', index='Feature', columns='Group', aggfunc='first', fill_value=0)
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_data.index)
    ax.set_title('特征热力图（绿色=开启，红色=关闭）')
    plt.colorbar(im, ax=ax)

else:  # 雷达图
    # 为雷达图准备数据（按组统计激活率）
    categories = list(st.session_state.features_data.keys())
    values = []
    for group_name, features in st.session_state.features_data.items():
        active_in_group = sum(1 for f in features if f['value'] == 1)
        total_in_group = len(features)
        values.append(active_in_group / total_in_group if total_in_group > 0 else 0)
    
    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, values, alpha=0.25, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_title('特征组激活率雷达图', size=14, pad=20)

plt.tight_layout()
st.pyplot(fig)

# 详细数据表格
st.markdown("---")
st.header("📋 详细数据表")

# 格式化数据表
display_df = df.copy()
display_df['Status'] = display_df.apply(
    lambda x: f"{'✅' if x['Status']=='safe' else '⚠️' if x['Status']=='warning' else '🔴'} {x['Status'].upper()}" 
    if x['Active'] else "⭕ INACTIVE", axis=1
)
display_df = display_df[['Feature', 'Value', 'Status', 'Group', 'Name']]

st.dataframe(display_df, use_container_width=True, hide_index=True)

# 底部总结（现在变量已经定义，不会报错）
st.markdown("---")
if danger_active > 0:
    st.error(f"🚨 **警告**: 当前有 {danger_active} 个危险特征处于激活状态（如高峰期），请谨慎监控！")
elif warning_active > 0:
    st.warning(f"⚠️ **注意**: 当前有 {warning_active} 个警告特征处于激活状态，系统运行正常但需关注。")
else:
    st.success(f"✅ **系统状态良好**: 所有激活特征均为安全状态，安全评分 {safety_score:.1f}%")

# 导出当前配置
st.markdown("---")
st.subheader("💾 导出配置")
config_text = "当前特征配置:\n"
for group_name, features in st.session_state.features_data.items():
    config_text += f"\n[{group_name}]\n"
    for f in features:
        config_text += f"{f['id']} = {f['value']}\n"

st.code(config_text, language='yaml')

if st.button("📋 复制配置到剪贴板"):
    st.write("配置已显示，请手动复制上方代码块内容")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
</style>
""", unsafe_allow_html=True)

# 页面标题
st.title("🚄 轨道交通特征预测分析系统")
st.markdown("---")

# 定义特征数据
features_data = {
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

# 主内容区域
st.header("📊 特征预测结果展示")

# 创建两列布局
col1, col2 = st.columns([2, 1])

with col1:
    # 展示特征分组
    for group_name, features in features_data.items():
        with st.container():
            st.subheader(f"🔹 {group_name}")
            
            for i, feature in enumerate(features):
                # 确定样式
                underline_class = "underline" if (feature.get("underline") and show_underline) else ""
                
                # 状态图标
                if feature["status"] == "safe":
                    icon = "✅"
                    status_class = "safe-status"
                elif feature["status"] == "warning":
                    icon = "⚠️"
                    status_class = "warning-status"
                else:
                    icon = "🔴"
                    status_class = "danger-status"
                
                # 显示特征
                st.markdown(f"""
                <div class="feature-item {underline_class}">
                    <b>{feature['id']} = {feature['value']}</b>: {feature['name']} 
                    <span class="{status_class}">{icon}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if show_metrics:
        st.subheader("📈 统计概览")
        
        # 计算统计数据
        total_features = sum(len(features) for features in features_data.values())
        safe_count = sum(1 for features in features_data.values() for f in features if f["status"] == "safe")
        warning_count = sum(1 for features in features_data.values() for f in features if f["status"] == "warning")
        danger_count = sum(1 for features in features_data.values() for f in features if f["status"] == "danger")
        
        # 显示指标卡片
        st.markdown(f"""
        <div class="metric-card">
            <h3>总特征数</h3>
            <h1>{total_features}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric(label="✅ 安全状态", value=safe_count, delta=f"{safe_count/total_features*100:.1f}%")
        st.metric(label="⚠️ 警告状态", value=warning_count, delta=f"{warning_count/total_features*100:.1f}%")
        st.metric(label="🔴 危险状态", value=danger_count, delta=f"{danger_count/total_features*100:.1f}%")

# 可视化图表区域
st.markdown("---")
st.header("📉 特征可视化分析")

# 准备数据用于图表
chart_data = []
for group_name, features in features_data.items():
    for feature in features:
        chart_data.append({
            'Group': group_name,
            'Feature': feature['id'],
            'Value': feature['value'],
            'Status': feature['status'],
            'Name': feature['name'][:30] + "..." if len(feature['name']) > 30 else feature['name']
        })

df = pd.DataFrame(chart_data)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

if chart_type == "条形图":
    colors = {'safe': '#4CAF50', 'warning': '#FF9800', 'danger': '#f44336'}
    bar_colors = [colors[status] for status in df['Status']]
    
    ax.barh(df['Feature'], df['Value'], color=bar_colors)
    ax.set_xlabel('特征值')
    ax.set_title('特征预测值分布')
    ax.set_xlim(0, 1.5)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['safe'], label='安全'),
                       Patch(facecolor=colors['warning'], label='警告'),
                       Patch(facecolor=colors['danger'], label='危险')]
    ax.legend(handles=legend_elements, loc='lower right')

elif chart_type == "热力图":
    # 创建透视表
    pivot_data = df.pivot_table(values='Value', index='Feature', columns='Group', aggfunc='first', fill_value=0)
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_data.index)
    ax.set_title('特征热力图')
    plt.colorbar(im, ax=ax)

else:  # 雷达图
    # 为雷达图准备数据
    categories = list(features_data.keys())
    values = [np.mean([f['value'] for f in features]) for features in features_data.values()]
    
    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, values, alpha=0.25, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1.5)
    ax.set_title('特征组雷达图', size=14, pad=20)

plt.tight_layout()
st.pyplot(fig)

# 详细数据表格
st.markdown("---")
st.header("📋 详细数据表")

# 格式化数据表
display_df = df.copy()
display_df['Status'] = display_df['Status'].map({
    'safe': '✅ 安全',
    'warning': '⚠️ 警告', 
    'danger': '🔴 危险'
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

# 底部总结
st.markdown("---")
st.success("🎯 **分析总结**: 当前系统大部分特征处于安全状态，但需注意高峰期(feature₁₅)和交通密度相关特征。")

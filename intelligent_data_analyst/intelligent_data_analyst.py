"""
智能数据分析助手
综合项目：Gradio + Pandas + LLM API
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import os
from dotenv import load_dotenv
import tempfile
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置区 ====================
load_dotenv()

api_key = os.getenv("LLM_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 全局变量：存储当前上传的数据
current_df = None


# ==================== 模块1：数据加载与EDA ====================
def process_file(file):
    """处理上传的CSV文件，返回数据概览"""
    global current_df
    
    if file is None:
        return None, None, "请先上传CSV文件", gr.update(choices=[]), gr.update(choices=[])
    
    try:
        current_df = pd.read_csv(file.name)
    except Exception as e:
        return None, None, f"❌ 读取失败: {str(e)}", gr.update(choices=[]), gr.update(choices=[])
    
    # 1. 数据前10行
    head_df = current_df.head(10)
    
    # 2. 描述统计
    try:
        desc_df = current_df.describe().reset_index()
    except:
        desc_df = pd.DataFrame({"提示": ["无非数值列可供描述统计"]})
    
    # 3. 缺失值和类型信息
    missing_info = []
    for col in current_df.columns:
        dtype = str(current_df[col].dtype)
        null_count = current_df[col].isnull().sum()
        null_pct = null_count / len(current_df) * 100
        missing_info.append(f"{col}: {dtype}, 缺失 {null_count} ({null_pct:.1f}%)")
    
    missing_text = "\n".join(missing_info)
    
    # 4. 更新下拉框选项
    all_cols = current_df.columns.tolist()
    numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
    
    return head_df, desc_df, missing_text, gr.update(choices=all_cols), gr.update(choices=numeric_cols)


# ==================== 模块2：AI智能问答 ====================
def ask_question(question):
    """基于数据摘要回答自然语言问题"""
    global current_df
    
    if current_df is None:
        return "⚠️ 请先上传CSV文件"
    
    if not question or not question.strip():
        return "⚠️ 请输入问题"
    
    # 构造数据摘要（截断防止超过API长度限制）
    summary_lines = []
    summary_lines.append(f"数据形状: {current_df.shape[0]} 行 × {current_df.shape[1]} 列")
    summary_lines.append("\n列名及数据类型:")
    
    for col in current_df.columns:
        dtype = str(current_df[col].dtype)
        unique = current_df[col].nunique()
        summary_lines.append(f"  - {col}: {dtype}, 唯一值 {unique} 个")
    
    # 数值列统计
    numeric_df = current_df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        summary_lines.append("\n数值列描述统计:")
        desc = numeric_df.describe().to_string()
        summary_lines.append(desc[:800])  # 截断
    
    # 类别列频数（前3）
    cat_cols = current_df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        summary_lines.append("\n类别列示例:")
        for col in cat_cols[:3]:
            top_vals = current_df[col].value_counts().head(3).to_string()
            summary_lines.append(f"  {col}:\n{top_vals}")
    
    # 前3行数据
    summary_lines.append(f"\n前3行数据:\n{current_df.head(3).to_string()}")
    
    data_summary = "\n".join(summary_lines)
    
    # 构造 Prompt
    prompt = f"""你是一位资深数据分析师，擅长从数据中发现洞察。
请基于以下CSV数据的摘要，回答用户的问题。

重要规则：
1. 只根据提供的数据摘要回答，绝对不要编造数据
2. 如果数据不足以回答，请明确说"根据现有数据无法确定"
3. 回答要简洁、专业，有逻辑
4. 如果涉及计算，请展示推理过程

数据摘要：
{data_summary[:2500]}  # 截断到2500字符防止超限

用户问题：{question}

请给出分析回答："""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ API调用出错: {str(e)}"


# ==================== 模块3：可视化引擎 ====================
def generate_plot(x_col, y_col, plot_type):
    """根据选择生成图表"""
    global current_df
    
    if current_df is None:
        return None
    
    if not x_col:
        return None
    
    plt.figure(figsize=(10, 6))
    
    try:
        if plot_type == "散点图":
            if y_col and y_col in current_df.columns:
                sns.scatterplot(data=current_df, x=x_col, y=y_col, s=100)
            else:
                return None
                
        elif plot_type == "柱状图":
            if y_col and y_col in current_df.columns:
                sns.barplot(data=current_df, x=x_col, y=y_col)
                plt.xticks(rotation=45)
            else:
                # 单变量柱状图：频数统计
                current_df[x_col].value_counts().head(20).plot(kind='bar')
                plt.xticks(rotation=45)
                
        elif plot_type == "折线图":
            if y_col and y_col in current_df.columns:
                plt.plot(current_df[x_col], current_df[y_col], marker='o')
            else:
                current_df[x_col].plot(marker='o')
                
        elif plot_type == "箱线图":
            if y_col and y_col in current_df.columns:
                sns.boxplot(data=current_df, x=x_col, y=y_col)
                plt.xticks(rotation=45)
            else:
                sns.boxplot(data=current_df, y=x_col)
                
        elif plot_type == "直方图":
            sns.histplot(current_df[x_col].dropna(), kde=True, bins=20)
            
        elif plot_type == "热力图":
            numeric_df = current_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
            else:
                return None
        
        title = f"{plot_type}: {x_col}"
        if y_col:
            title += f" vs {y_col}"
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        # 保存到临时文件
        tmp_path = os.path.join(tempfile.gettempdir(), "gradio_plot.png")
        plt.savefig(tmp_path, dpi=150, bbox_inches="tight")
        plt.close()
        return tmp_path
        
    except Exception as e:
        plt.close()
        return None


# ==================== 模块4：自动可视化（AI推荐） ====================
def auto_visualize():
    """自动生成3张关键图表"""
    global current_df
    
    if current_df is None:
        return None, None, None
    
    plots = []
    
    try:
        # 图1：数值分布直方图（第一个数值列）
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(current_df[numeric_cols[0]].dropna(), kde=True, color="steelblue")
            plt.title(f"{numeric_cols[0]} 分布")
            plt.tight_layout()
            p1 = os.path.join(tempfile.gettempdir(), "auto1.png")
            plt.savefig(p1, dpi=150)
            plt.close()
            plots.append(p1)
        else:
            plots.append(None)
        
        # 图2：类别频数（第一个类别列）
        cat_cols = current_df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            plt.figure(figsize=(8, 5))
            current_df[cat_cols[0]].value_counts().head(10).plot(kind='barh', color="coral")
            plt.title(f"{cat_cols[0]} 频数 TOP10")
            plt.tight_layout()
            p2 = os.path.join(tempfile.gettempdir(), "auto2.png")
            plt.savefig(p2, dpi=150)
            plt.close()
            plots.append(p2)
        else:
            plots.append(None)
        
        # 图3：相关性热力图（如果有2个以上数值列）
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(8, 6))
            sns.heatmap(current_df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("数值特征相关性热力图")
            plt.tight_layout()
            p3 = os.path.join(tempfile.gettempdir(), "auto3.png")
            plt.savefig(p3, dpi=150)
            plt.close()
            plots.append(p3)
        else:
            plots.append(None)
            
        return plots[0] if len(plots) > 0 else None, \
               plots[1] if len(plots) > 1 else None, \
               plots[2] if len(plots) > 2 else None
               
    except Exception as e:
        return None, None, None


# ==================== Gradio界面搭建 ====================
with gr.Blocks(title="智能数据分析助手", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🤖 智能数据分析助手
    上传CSV文件，AI自动完成数据探索、智能问答和可视化。
    """)
    
    # 文件上传区
    with gr.Row():
        file_input = gr.File(label="📁 上传CSV文件", file_types=[".csv"])
        upload_btn = gr.Button("🔍 开始分析", variant="primary")
    
    # 标签页
    with gr.Tabs():
        
        # Tab 1: 数据概览
        with gr.TabItem("📊 数据概览"):
            gr.Markdown("### 数据样本（前10行）")
            head_output = gr.Dataframe()
            
            gr.Markdown("### 描述统计")
            desc_output = gr.Dataframe()
            
            gr.Markdown("### 数据质量报告")
            missing_output = gr.Textbox(lines=8, label="列类型与缺失值")
            
            gr.Markdown("### AI自动可视化")
            with gr.Row():
                auto_plot1 = gr.Image(label="分布图")
                auto_plot2 = gr.Image(label="频数图")
                auto_plot3 = gr.Image(label="相关性")
            
            auto_viz_btn = gr.Button("✨ 生成自动可视化")
        
        # Tab 2: 智能问答
        with gr.TabItem("💬 智能问答"):
            gr.Markdown("""
            ### 用自然语言向AI提问
            示例问题：
            - "哪个月销量最高？"
            - "哪个区域的平均评分最高？"
            - "销量和评分之间有关系吗？"
            """)
            question_input = gr.Textbox(
                label="你的问题",
                placeholder="输入自然语言问题，AI基于数据回答...",
                lines=3
            )
            ask_btn = gr.Button("🚀 提问", variant="primary")
            answer_output = gr.Textbox(
                label="AI分析结果",
                lines=12,
            )
        
        # Tab 3: 自定义可视化
        with gr.TabItem("📈 自定义可视化"):
            gr.Markdown("### 选择列生成图表")
            with gr.Row():
                x_dropdown = gr.Dropdown(label="X轴 / 主要列", choices=[], allow_custom_value=True)
                y_dropdown = gr.Dropdown(label="Y轴 / 次要列（可选）", choices=[], allow_custom_value=True)
            
            plot_type = gr.Radio(
                ["散点图", "柱状图", "折线图", "箱线图", "直方图", "热力图"],
                label="图表类型",
                value="散点图"
            )
            
            plot_btn = gr.Button("📊 生成图表")
            plot_output = gr.Image(label="图表结果")
    
    # ==================== 事件绑定 ====================
    
    # 上传按钮：分析文件
    upload_btn.click(
        fn=process_file,
        inputs=file_input,
        outputs=[head_output, desc_output, missing_output, x_dropdown, y_dropdown]
    )
    
    # 自动可视化按钮
    auto_viz_btn.click(
        fn=auto_visualize,
        inputs=None,
        outputs=[auto_plot1, auto_plot2, auto_plot3]
    )
    
    # 问答按钮
    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )
    
    # 可视化按钮
    plot_btn.click(
        fn=generate_plot,
        inputs=[x_dropdown, y_dropdown, plot_type],
        outputs=plot_output
    )


# ==================== 启动 ====================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,        # 设为 True 可生成公网链接（24小时有效）
        show_error=True
    )
    print("🚀 应用已启动，请访问 http://127.0.0.1:7860")

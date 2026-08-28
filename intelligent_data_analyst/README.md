{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "84fb32fa-e26a-4f57-8e47-711a30f521fa",
   "metadata": {},
   "source": [
    "# 🤖 智能数据分析助手\n",
    "\n",
    "## 项目简介\n",
    "这是一个基于 Gradio + LLM API 的智能数据分析平台。用户上传 CSV 文件后，系统自动完成数据探索、生成可视化，并支持自然语言问答。\n",
    "\n",
    "## 功能特性\n",
    "- 📊 **数据概览**：自动展示数据样本、描述统计、缺失值报告\n",
    "- 💬 **智能问答**：用自然语言提问，AI 基于数据内容回答\n",
    "- 📈 **一键可视化**：自动生成分布图、频数图、相关性热力图\n",
    "- 🎨 **自定义图表**：支持散点图、柱状图、折线图、箱线图、直方图、热力图\n",
    "\n",
    "## 技术栈\n",
    "| 模块 | 技术 |\n",
    "|------|------|\n",
    "| 前端界面 | Gradio |\n",
    "| 数据处理 | Pandas, NumPy |\n",
    "| 可视化 | Matplotlib, Seaborn |\n",
    "| AI 引擎 | DeepSeek API (Function Calling / Prompt Engineering) |\n",
    "\n",
    "## 使用流程"
   ]
  },
  {
   "cell_type": "raw",
   "id": "12cba984-8158-49b4-8f7e-4959a0be6dde",
   "metadata": {},
   "source": [
    "上传CSV → 自动EDA → 智能问答 / 自定义可视化"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99a87700-a5ba-47d6-ac92-777257456ea3",
   "metadata": {},
   "source": [
    "\n",
    "## 与其他项目的联系\n",
    "- (EDA): 数据概览、描述统计、缺失值分析\n",
    "- (LLM API): 调用 DeepSeek 进行自然语言理解\n",
    "- (可视化): Matplotlib/Seaborn 图表生成\n",
    "- (Prompt工程): 构造数据摘要 Prompt，约束 AI 不编造数据"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

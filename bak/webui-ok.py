# prompt_app_final_fixed.py 


import os
import openai
import gradio as gr
import re
import json
from dotenv import load_dotenv



# --- 1. 初期設定 ---
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY_TEXT')
if not openai.api_key:
    print("エラー: OpenAIのAPIキーが設定されていません。.envファイルを確認してください。")

# --- 2. コア機能の関数化 ---

def refine_prompt_with_feedback(base_keywords, user_feedback):
    """フィードバックを元にプロンプトを改良する関数"""
    system_prompt = (
        "あなたは、ユーザーのフィードバックを元にStable Diffusionのプロンプトを改良する専門家です。"
        "「ベースとなるキーワード」と「ユーザーの要望」を組み合わせ、最適な新しいプロンプトを生成してください。"
        "出力は、「日本語」「英語」「キーワード」の3つの要素を含むフォーマットで返してください。"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"ベースとなるキーワード: {base_keywords}\n\nユーザーの要望: {user_feedback}"}],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"フィードバックの反映に失敗しました: {e}"

def optimize_initial_prompt(user_prompt):
    """最初の入力を詳細なプロンプトに変換する関数"""
    system_prompt = (
        "あなたは、入力された簡単なテーマを、Stable Diffusionでの画像生成に適した詳細な英語のプロンプトに変換する専門家です。"
        "出力には、詳細化した日本語の説明、英語の翻訳、そしてカンマ区切りの英語キーワードの3つを以下のフォーマットで記述してください。\n"
        "フォーマット:\n日本語: [ここに説明]\n英語: [ここに翻訳]\nキーワード: [ここにキーワード]"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"テーマ: {user_prompt}"}], max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"プロンプトの最適化に失敗しました: {e}"

def extract_keywords(full_prompt):
    """フル出力からキーワード部分のみを抽出する関数"""
    match = re.search(r"キーワード:\s*(.*)", full_prompt, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "キーワードが見つかりませんでした。"

# --- 3. 改良されたログ表示のための新関数 ---
def format_history_log(log_data):
    """構造化されたログデータを、見やすいテキスト形式に変換する関数"""
    if not log_data:
        return ""
    
    formatted_string = ""
    for i, item in enumerate(log_data):
        step_header = f"--- ステップ {i+1}: {item['type']} ---\n"
        formatted_string += step_header
        
        # ユーザーの入力内容を表示
        if item.get('input'):
            # ★★★ ここが修正点 ★★★
            # f-stringの外で置換処理を行う
            user_input_formatted = item['input'].replace('\n', '\n> ')
            formatted_string += f"あなたの入力:\n> {user_input_formatted}\n\n"
            
        # 生成されたプロンプトを表示
        formatted_string += f"生成されたプロンプト:\n{item['output_prompt']}\n\n"
        
    return formatted_string.strip()

# --- 4. Gradioアプリケーションの構築 ---

with gr.Blocks(theme=gr.themes.Default(),title="対話型プロンプトナビゲーション") as demo:
    
    structured_history_log = gr.State([])
    current_keywords_state = gr.State("")

    gr.Markdown("# 対話型プロンプトナビゲーション (ログ改良版)")
    gr.Markdown(
        "**AIと対話しながら、画像生成プロンプトを共同で作成・改良するツールです。**\n"
        "下のテキストボックスに追加の指示を入力し、「プロンプトを改良」ボタンを押してください。"
    )

    with gr.Column():
        with gr.Row():
            initial_prompt_input = gr.Textbox(label="1. 生成したい画像のテーマを入力", placeholder="例：雨の日に傘をさす猫", scale=3)
            start_button = gr.Button("ナビゲーション開始", variant="primary", scale=1)
        
        gr.Markdown("---")
        current_prompt_display = gr.Markdown(label="現在のプロンプト詳細")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 画像生成ツール用キーワード")
                copyable_prompt_textbox = gr.Textbox(label="編集可能", show_copy_button=True, interactive=True)
            
            with gr.Column(scale=1):
                gr.Markdown("### 2. プロンプトを改良する")
                feedback_input = gr.Textbox(label="追加の指示を入力", placeholder="例：もっと猫をふわふわにして。背景を夜に変えて。")
                refine_button = gr.Button("プロンプトを改良", variant="primary")
        
        gr.Markdown("---")
        history_display = gr.Textbox(label="ナビゲーション履歴", lines=15, interactive=False)
        feedback_text = gr.Textbox(label="フィードバック", interactive=False)

    # --- 5. UIのイベントハンドリング ---

    def update_all_components(new_full_prompt, new_log_entry, history_log):
        """UI更新と状態管理をまとめた関数"""
        current_history = history_log + [new_log_entry]
        new_keywords = extract_keywords(new_full_prompt)
        
        history_text = format_history_log(current_history)
        
        return {
            current_prompt_display: new_full_prompt,
            copyable_prompt_textbox: new_keywords,
            structured_history_log: current_history,
            history_display: history_text,
            current_keywords_state: new_keywords,
            feedback_input: ""
        }

    def start_navigation(initial_prompt):
        """「ナビゲーション開始」ボタンの処理"""
        initial_full_prompt = optimize_initial_prompt(initial_prompt)
        log_entry = {"type": "初期テーマ", "input": initial_prompt, "output_prompt": initial_full_prompt}
        updates = update_all_components(initial_full_prompt, log_entry, [])
        updates[feedback_text] = "初期プロンプトを生成しました。追加の指示を入力して改良してください。"
        return updates

    def handle_refinement(user_feedback, history_log, current_keywords, manual_keywords):
        """「プロンプトを改良」ボタンの処理"""
        # 手動編集されたキーワードがある場合、それを使用
        keywords_to_use = manual_keywords if manual_keywords else current_keywords
        
        if not user_feedback:
            return {feedback_text: "追加の指示を入力してください。"}

        log_type = "自由な言葉で指示"
        log_input = user_feedback
        new_full_prompt = refine_prompt_with_feedback(keywords_to_use, user_feedback)
        feedback_msg = "あなたの指示を元にプロンプトを更新しました。"

        log_entry = {"type": log_type, "input": log_input, "output_prompt": new_full_prompt}
        updates = update_all_components(new_full_prompt, log_entry, history_log)
        updates[feedback_text] = feedback_msg
        return updates

    outputs_list = [
        current_prompt_display, copyable_prompt_textbox, feedback_text,
        structured_history_log, history_display, current_keywords_state,
        feedback_input
    ]

    start_button.click(fn=start_navigation, inputs=[initial_prompt_input], outputs=outputs_list)
    refine_button.click(fn=handle_refinement, inputs=[feedback_input, structured_history_log, current_keywords_state, copyable_prompt_textbox], outputs=outputs_list)

# --- 6. アプリケーションの起動 ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",server_port=7878, debug=True)
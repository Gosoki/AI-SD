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

def get_neighbor_prompts_with_translation(current_keywords, history_log):
    """和訳付きの隣接ノード（改良案）を生成する関数"""
    history_keywords = [extract_keywords(item['output_prompt']) for item in history_log]
    history_str = "\n".join(f"- {kw}" for kw in history_keywords)
    
    system_prompt = (
        "あなたは、画像生成プロンプトの優秀なナビゲーターです。"
        "現在の英語キーワードを元に、単語を1つ追加・削除・変更する形で、5つの異なる改良案を提案してください。"
        f"これまでの履歴にあるプロンプトは提案しないでください:\n{history_str}\n"
        '出力は、以下のキーを持つオブジェクトのリストを含むJSON形式で返してください: {"suggestions": [{"english_keywords": "...", "japanese_translation": "..."}]}。'
        "他の説明は一切不要です。"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"現在のキーワード: {current_keywords}"}],
            max_tokens=500, temperature=0.7, response_format={"type": "json_object"}
        )
        content = json.loads(response.choices[0].message["content"])
        return content.get("suggestions", [])
    except Exception as e:
        print(f"GPT提案生成エラー: {e}")
        return [{"english_keywords": "エラー", "japanese_translation": f"提案の生成に失敗: {e}"}]

def refine_prompt_with_feedback(base_keywords, user_feedback):
    """フィードバックを元にプロンプトを改良する関数"""
    system_prompt = (
        "あなたは、ユーザーのフィードバックを元にStable Diffusionのプロンプトを改良する専門家です。"
        "「ベースとなるキーワード」と「ユーザーの要望」を組み合わせ、最適な新しいプロンプトを生成してください。"
        "出力は、「日本語」「英語」「キーワード」の3つの要素を含むフォーマットで返してください。"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"ベースとなるキーワード: {base_keywords}\n\nユーザーの要望: {user_feedback}"}],
            max_tokens=300
        )
        return response.choices[0].message["content"].strip()
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
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"テーマ: {user_prompt}"}], max_tokens=300
        )
        return response.choices[0].message["content"].strip()
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

with gr.Blocks(theme=gr.themes.Soft(),title="対話型プロンプトナビゲーション") as demo:
    
    structured_history_log = gr.State([])
    current_keywords_state = gr.State("")

    gr.Markdown("# 対話型プロンプトナビゲーション (ログ改良版)")
    gr.Markdown(
        "**AIと対話しながら、画像生成プロンプトを共同で作成・改良するツールです。**\n"
        "下の**AかB、または両方を組み合わせて**プロンプトを改良し、「プロンプトを改良」ボタンを押してください。"
    )

    with gr.Column():
        with gr.Row():
            initial_prompt_input = gr.Textbox(label="1. 生成したい画像のテーマを入力", placeholder="例：雨の日に傘をさす猫", scale=3)
            start_button = gr.Button("ナビゲーション開始", variant="primary", scale=1)
        
        gr.Markdown("---")
        current_prompt_display = gr.Markdown(label="現在のプロンプト詳細")
        copyable_prompt_textbox = gr.Textbox(label="画像生成ツール用キーワード（コピーしてお使いください）", show_copy_button=True)
        
        gr.Markdown("---")
        gr.Markdown("### 2. プロンプトを改良する")
        neighbor_prompts_radio = gr.Radio(label="A. AIからの改良案（任意選択）", choices=[], interactive=True)
        feedback_input = gr.Textbox(label="B. または、自由な言葉で追加の指示を入力", placeholder="例：もっと猫をふわふわにして。背景を夜に変えて。")
        hybrid_refine_button = gr.Button("プロンプトを改良", variant="primary")
        
        gr.Markdown("---")
        history_display = gr.Textbox(label="ナビゲーション履歴", lines=15, interactive=False)
        feedback_text = gr.Textbox(label="フィードバック", interactive=False)

    # --- 5. UIのイベントハンドリング ---

    def update_all_components(new_full_prompt, new_log_entry, history_log):
        """UI更新と状態管理をまとめた関数"""
        current_history = history_log + [new_log_entry]
        new_keywords = extract_keywords(new_full_prompt)
        
        neighbor_suggestions = get_neighbor_prompts_with_translation(new_keywords, current_history)
        radio_choices = [(f"{item['english_keywords']} ({item['japanese_translation']})", item['english_keywords']) for item in neighbor_suggestions]
        
        history_text = format_history_log(current_history)
        
        return {
            current_prompt_display: new_full_prompt,
            copyable_prompt_textbox: new_keywords,
            structured_history_log: current_history,
            history_display: history_text,
            neighbor_prompts_radio: gr.update(choices=radio_choices, value=None),
            current_keywords_state: new_keywords,
            feedback_input: ""
        }

    def start_navigation(initial_prompt):
        """「ナビゲーション開始」ボタンの処理"""
        initial_full_prompt = optimize_initial_prompt(initial_prompt)
        log_entry = {"type": "初期テーマ", "input": initial_prompt, "output_prompt": initial_full_prompt}
        updates = update_all_components(initial_full_prompt, log_entry, [])
        updates[feedback_text] = "初期プロンプトを生成しました。AかB、または両方を使って改良してください。"
        return updates

    def handle_hybrid_refinement(selected_suggestion, user_feedback, history_log, current_keywords):
        """「プロンプトを改良」ボタンのハイブリッド処理"""
        new_full_prompt, log_entry, feedback_msg = "", {}, ""
        if selected_suggestion and user_feedback:
            log_type = "提案選択 ＋ 自由指示"
            log_input = f"提案: {selected_suggestion}\n指示: {user_feedback}"
            new_full_prompt = refine_prompt_with_feedback(selected_suggestion, user_feedback)
            feedback_msg = "選択した提案に、さらにあなたの指示を加えてプロンプトを更新しました。"
        elif selected_suggestion:
            log_type = "提案を選択"
            log_input = selected_suggestion
            new_full_prompt = optimize_initial_prompt(selected_suggestion)
            feedback_msg = "選択した提案を元にプロンプトを更新しました。"
        elif user_feedback:
            log_type = "自由な言葉で指示"
            log_input = user_feedback
            new_full_prompt = refine_prompt_with_feedback(current_keywords, user_feedback)
            feedback_msg = "あなたの指示を元にプロンプトを更新しました。"
        else:
            return {feedback_text: "改良方法（AまたはB）を入力してください。"}

        log_entry = {"type": log_type, "input": log_input, "output_prompt": new_full_prompt}
        updates = update_all_components(new_full_prompt, log_entry, history_log)
        updates[feedback_text] = feedback_msg
        return updates

    outputs_list = [
        current_prompt_display, copyable_prompt_textbox, feedback_text,
        structured_history_log, history_display, neighbor_prompts_radio, current_keywords_state,
        feedback_input
    ]

    start_button.click(fn=start_navigation, inputs=[initial_prompt_input], outputs=outputs_list)
    hybrid_refine_button.click(fn=handle_hybrid_refinement, inputs=[neighbor_prompts_radio, feedback_input, structured_history_log, current_keywords_state], outputs=outputs_list)

# --- 6. アプリケーションの起動 ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",server_port=7878, debug=True)
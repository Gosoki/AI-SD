import os
import openai
import gradio as gr
import re
import requests
import base64
import io
import json
from PIL import Image
from dotenv import load_dotenv

# --- 1. 初期設定 ---
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY_TEXT')
SD_API_URL = os.getenv('STABLE_DIFFUSION_API_URL')
TAGGER_API_URL = os.getenv('TAGGER_API_URL',SD_API_URL)
if not openai.api_key:
    print("エラー: OpenAIのAPIキーが設定されていません。.envファイルを確認してください。")
if not SD_API_URL:
    print("エラー: Stable DiffusionのAPI URLが設定されていません。.envファイルを確認してください。")
print(SD_API_URL)

# --- 2. コア機能の関数化 ---
def refine_prompt_with_feedback(base_keywords, negative_keywords, user_feedback):
    """フィードバックを元にプロンプトを改良する関数"""
    system_prompt = (
        "あなたは、ユーザーのフィードバックを元にStable Diffusionのプロンプトを改良する専門家です。"
        "「ベースとなるキーワード」と「ユーザーの要望」を組み合わせ、最適な新しいプロンプトを生成してください。"
        "「ネガティブキーワード」も同様に改良してください。"
        "请重新按需求生成新的prompt，可以参考之前的用输入，但是按总要程度重新排序"
        "注意：不要只是在现有关键词后添加新内容，而是根据用户需求重新生成一套完整的关键词"
        "出力は、「日本語」「英語」「キーワード」「ネガティブキーワード」の4つの要素を含むフォーマットで返してください。\n"
        "フォーマット:\n日本語: [ここに説明]\n英語: [ここに翻訳]\nキーワード: [ここにキーワード]\nネガティブキーワード: [ここに含めたくない要素]"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"ベースとなるキーワード: {base_keywords}\nネガティブキーワード: {negative_keywords}\n\nユーザーの要望: {user_feedback}"}],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"フィードバックの反映に失敗しました: {e}"

def optimize_initial_prompt(user_prompt):
    """最初の入力を詳細なプロンプトに変換する関数"""
    system_prompt = (
        "あなたは、入力された簡単なテーマを、Stable Diffusionでの画像生成に適した詳細な英語のプロンプトに変換する専門家です。"
        "出力には、詳細化した日本語の説明、英語の翻訳、カンマ区切りの英語キーワード、そして画像に含めたくない要素（ネガティブプロンプト）の4つを以下のフォーマットで記述してください。\n"
        "フォーマット:\n日本語: [ここに説明]\n英語: [ここに翻訳]\nキーワード: [ここにキーワード]\nネガティブキーワード: [ここに含めたくない要素]"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"テーマ: {user_prompt}"}], max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"プロンプトの最適化に失敗しました: {e}"

def extract_keywords(full_prompt):
    """フル出力からキーワード部分のみを抽出する関数"""
    match = re.search(r"キーワード:\s*(.*?)(?:\nネガティブキーワード:|$)", full_prompt, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "キーワードが見つかりませんでした。"

def extract_negative_keywords(full_prompt):
    """フル出力からネガティブキーワード部分のみを抽出する関数"""
    match = re.search(r"ネガティブキーワード:\s*(.*?)(?:\n|$)", full_prompt, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "blurry, low quality"

def generate_image(keywords, negative_keywords):
    """キーワードを使用して画像を生成する関数"""
    try:
        url = SD_API_URL+"/sdapi/v1/txt2img"
        print(url)
        payload = {
            "prompt": keywords,
            "negative_prompt": negative_keywords+"blurry, low quality",
            "steps": 27,
            "sampler_index": "DPM++ 2M",
            "width": 512,
            "height": 512,
            "cfg_scale": 7.0,
            "seed": -1
        }
        
        response = requests.post(url, json=payload)
        r = response.json()
        
        # Base64エンコードされた画像をデコード
        image_data = base64.b64decode(r['images'][0])
        
        # PILイメージとして読み込む
        image = Image.open(io.BytesIO(image_data))
        
        return image
    except Exception as e:
        print(f"画像生成エラー: {e}")
        return None

def interrogate_image(image, model="wd14-vit-v2-git", threshold=0.4):
    """画像からタグを抽出する関数"""
    if image is None:
        return "画像が提供されていません。"
    
    try:
        # 画像をバイト列に変換
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # APIリクエストを構築
        interrogate_endpoint = SD_API_URL + "/tagger/v1/interrogate"
        payload = {
            "image": img_b64,
            "model": model,
            "threshold": threshold
        }
        headers = {
            "Content-Type": "application/json"
        }
        print(interrogate_endpoint, model, threshold)
        # APIリクエストを送信
        response = requests.post(interrogate_endpoint, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            caption = data.get("caption", {})
            
            # Gradio Labelコンポーネント用のフォーマットに変換
            if isinstance(caption, dict):
                # 除外するタグのリスト
                exclude_tags = ['general', 'sensitive', 'questionable', 'explicit']
                
                # 除外タグを削除
                filtered_caption = {k: v for k, v in caption.items() if k not in exclude_tags}
                
                # 信頼度でソート
                sorted_tags = sorted(filtered_caption.items(), key=lambda x: x[1], reverse=True)
                
                # Label用の辞書を作成
                label_dict = {tag: float(confidence) for tag, confidence in sorted_tags}
                
                # タグを日本語に翻訳
                translated_dict = translate_tags_to_japanese(label_dict)
                return translated_dict
            
            # 辞書でない場合は空の辞書を返す
            return {}
        else:
            return {}
    except Exception as e:
        print(f"タグ抽出エラー: {str(e)}")
        return {}

def translate_tags_to_japanese(tags_dict):
    """英語のタグを日本語に翻訳する関数"""
    if not tags_dict or len(tags_dict) == 0:
        return {}
    
    try:
        # 翻訳するタグのリストを作成（上位20個まで）
        tags_to_translate = list(tags_dict.keys())
        
        # タグが1つもない場合は空の辞書を返す
        if not tags_to_translate:
            return {}
        
        # OpenAI APIを使用して翻訳
        system_prompt = "あなたは翻訳の専門家です。与えられた英語のタグリストを日本語に翻訳してください。元の英語のタグと日本語訳のペアをJSON形式で返してください。"
        user_prompt = f"以下のタグを日本語に翻訳してください。簡潔に翻訳し、余計な説明は不要です:\n{', '.join(tags_to_translate)}"
        
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        # レスポンスからJSONを抽出
        translation_text = response.choices[0].message.content
        
        # JSONをパース
        try:
            import json
            translations = json.loads(translation_text)
            
            # 元のタグ辞書と翻訳を組み合わせる
            result = {}
            for tag, confidence in tags_dict.items():
                if tag in translations:
                    # 英語タグ（日本語訳）の形式で表示
                    jp_tag = translations[tag]
                    new_tag = f"{tag} ({jp_tag})"
                    result[new_tag] = confidence
                else:
                    result[tag] = confidence
            
            return result
        except json.JSONDecodeError:
            print("JSONのパースに失敗しました")
            return tags_dict
    except Exception as e:
        print(f"翻訳エラー: {str(e)}")
        return tags_dict

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

# --- 5. UIのイベントハンドリング ---

def update_all_components(new_full_prompt, new_log_entry, history_log):
    """UI更新と状態管理をまとめた関数"""
    current_history = history_log + [new_log_entry]
    new_keywords = extract_keywords(new_full_prompt)
    new_negative_keywords = extract_negative_keywords(new_full_prompt)
    
    history_text = format_history_log(current_history)
    
    return {
        current_prompt_display: new_full_prompt,
        copyable_prompt_textbox: new_keywords,
        negative_prompt_textbox: new_negative_keywords,
        structured_history_log: current_history,
        history_display: history_text,
        current_keywords_state: new_keywords,
        current_negative_keywords_state: new_negative_keywords,
        feedback_input: ""
    }

def start_navigation(initial_prompt):
    """「ナビゲーション開始」ボタンの処理"""
    initial_full_prompt = optimize_initial_prompt(initial_prompt)
    log_entry = {"type": "初期テーマ", "input": initial_prompt, "output_prompt": initial_full_prompt}
    updates = update_all_components(initial_full_prompt, log_entry, [])
    updates[feedback_text] = "初期プロンプトを生成しました。追加の指示を入力して改良してください。"
    return updates

def handle_refinement(user_feedback, history_log, current_keywords, current_negative_keywords, manual_keywords, manual_negative_keywords):
    """「プロンプトを改良」ボタンの処理"""
    # 手動編集されたキーワードがある場合、それを使用
    keywords_to_use = manual_keywords if manual_keywords else current_keywords
    negative_keywords_to_use = manual_negative_keywords if manual_negative_keywords else current_negative_keywords
    
    if not user_feedback:
        return {feedback_text: "追加の指示を入力してください。"}

    log_type = "自由な言葉で指示"
    log_input = user_feedback
    new_full_prompt = refine_prompt_with_feedback(keywords_to_use, negative_keywords_to_use, user_feedback)
    feedback_msg = "あなたの指示を元にプロンプトを更新しました。"

    log_entry = {"type": log_type, "input": log_input, "output_prompt": new_full_prompt}
    updates = update_all_components(new_full_prompt, log_entry, history_log)
    updates[feedback_text] = feedback_msg
    return updates

def handle_image_generation(keywords, negative_keywords):
    """「画像を生成」ボタンの処理"""
    if not keywords:
        return {feedback_text: "キーワードが入力されていません。", generated_image: None, generated_image_state: None, extracted_tags_label: None, tags_state: {}, selected_tags: []}
    
    try:
        image = generate_image(keywords, negative_keywords)
        if image:
            # 画像生成後、自動的にタグを抽出
            try:
                model = "wd14-vit-v2-git"
                threshold = 0.4
                tags = interrogate_image(image, model, threshold)
                
                # タグの選択肢を更新
                tag_choices = list(tags.keys()) if tags and len(tags) > 0 else []
                
                return {
                    feedback_text: "画像を生成し、タグを抽出しました。", 
                    generated_image: image, 
                    generated_image_state: image,
                    extracted_tags_label: tags if tags and len(tags) > 0 else None,
                    tags_state: tags,
                    selected_tags: gr.update(choices=tag_choices, value=[])
                }
            except Exception as e:
                print(f"自動タグ抽出エラー: {str(e)}")
                return {
                    feedback_text: "画像を生成しましたが、タグ抽出に失敗しました。", 
                    generated_image: image, 
                    generated_image_state: image,
                    extracted_tags_label: None,
                    tags_state: {},
                    selected_tags: []
                }
        else:
            return {
                feedback_text: "画像の生成に失敗しました。", 
                generated_image: None, 
                generated_image_state: None,
                extracted_tags_label: None,
                tags_state: {},
                selected_tags: []
            }
    except Exception as e:
        return {
            feedback_text: f"エラー: {str(e)}", 
            generated_image: None, 
            generated_image_state: None,
            extracted_tags_label: None,
            tags_state: {},
            selected_tags: []
        }

def handle_tag_extraction(image):
    """「タグを抽出」ボタンの処理"""
    if image is None:
        return {
            feedback_text: "画像が生成されていません。先に画像を生成してください。", 
            extracted_tags_label: None,
            tags_state: {},
            selected_tags: []
        }
    
    try:
        # 固定のモデルと閾値を使用
        model = "wd14-vit-v2-git"
        threshold = 0.4
        tags = interrogate_image(image, model, threshold)
        
        if tags and len(tags) > 0:
            # タグの選択肢を更新
            tag_choices = list(tags.keys())
            
            return {
                feedback_text: "タグを抽出しました。", 
                extracted_tags_label: tags,
                tags_state: tags,
                selected_tags: gr.update(choices=tag_choices, value=[])
            }
        else:
            return {
                feedback_text: "タグが見つかりませんでした。", 
                extracted_tags_label: None,
                tags_state: {},
                selected_tags: []
            }
    except Exception as e:
        return {
            feedback_text: f"タグ抽出エラー: {str(e)}", 
            extracted_tags_label: None,
            tags_state: {},
            selected_tags: []
        }

def add_selected_tags_to_prompt(selected_tag_names, current_prompt):
    """選択したタグをプロンプトに追加する関数"""
    if not selected_tag_names or len(selected_tag_names) == 0:
        return {feedback_text: "タグが選択されていません。", copyable_prompt_textbox: current_prompt}
    
    # タグから余分な情報を削除 (例: "tag (翻訳)" -> "tag")
    clean_tags = []
    for tag in selected_tag_names:
        # 括弧がある場合は括弧の前の部分だけを取る
        if " (" in tag:
            clean_tag = tag.split(" (")[0].strip()
            clean_tags.append(clean_tag)
        else:
            clean_tags.append(tag)
    
    # 現在のプロンプトにタグを追加
    if current_prompt and current_prompt.strip():
        new_prompt = current_prompt.strip() + ", " + ", ".join(clean_tags)
    else:
        new_prompt = ", ".join(clean_tags)
    
    return {
        feedback_text: "選択したタグをプロンプトに追加しました。", 
        copyable_prompt_textbox: new_prompt
    }

def add_selected_tags_to_negative_prompt(selected_tag_names, current_negative_prompt):
    """選択したタグをネガティブプロンプトに追加する関数"""
    if not selected_tag_names or len(selected_tag_names) == 0:
        return {feedback_text: "タグが選択されていません。", negative_prompt_textbox: current_negative_prompt}
    
    # タグから余分な情報を削除 (例: "tag (翻訳)" -> "tag")
    clean_tags = []
    for tag in selected_tag_names:
        # 括弧がある場合は括弧の前の部分だけを取る
        if " (" in tag:
            clean_tag = tag.split(" (")[0].strip()
            clean_tags.append(clean_tag)
        else:
            clean_tags.append(tag)
    
    # 現在のネガティブプロンプトにタグを追加
    if current_negative_prompt and current_negative_prompt.strip():
        new_prompt = current_negative_prompt.strip() + ", " + ", ".join(clean_tags)
    else:
        new_prompt = ", ".join(clean_tags)
    
    return {
        feedback_text: "選択したタグをネガティブプロンプトに追加しました。", 
        negative_prompt_textbox: new_prompt
    }


# --- 4. Gradioアプリケーションの構築 ---
with gr.Blocks(theme=gr.themes.Soft(),title="対話型プロンプトナビゲーション") as demo:
    structured_history_log = gr.State([])
    current_keywords_state = gr.State("")
    current_negative_keywords_state = gr.State("")
    generated_image_state = gr.State(None)
    tags_state = gr.State({})

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
                copyable_prompt_textbox = gr.Textbox(label="編集可能", show_copy_button=True, interactive=True, lines=5)
                negative_prompt_textbox = gr.Textbox(label="ネガティブキーワード（編集可能）", show_copy_button=True, interactive=True, lines=5)
                
            
            with gr.Column(scale=1):
                gr.Markdown("### 2. プロンプトを改良する")
                feedback_input = gr.Textbox(label="追加の指示を入力", placeholder="例：もっと猫をふわふわにして。背景を夜に変えて。", lines=8)
                refine_button = gr.Button("プロンプトを改良", variant="primary", size="lg", scale=1)
        
        gr.Markdown("---")
        generate_button = gr.Button("画像を生成", variant="primary", size="lg", scale=2)
        with gr.Row():

            with gr.Column(scale=1):
                
                
                # 新しく追加：タグ選択用のチェックボックス
                selected_tags = gr.CheckboxGroup(
                    label="タグを選択",
                    choices=[],
                    interactive=True
                )
                
                with gr.Row():
                    # 選択したタグを追加するボタン
                    add_tags_button = gr.Button("選択したタグを提示語に追加", variant="secondary")
                    
                    # 選択したタグを負面提示語に追加するボタン
                    add_negative_tags_button = gr.Button("選択したタグを負面提示語に追加", variant="secondary")
                    
                extracted_tags_label = gr.Label(
                    label="抽出されたタグ",
                    show_label=True,
                    num_top_classes=30,
                    scale=2
                )
                
            with gr.Column(scale=1):
                
                generated_image = gr.Image(label="生成された画像", type="pil")
                tagger_button = gr.Button("タグを抽出", variant="secondary")
                
        gr.Markdown("---")        
        feedback_text = gr.Textbox(label="フィードバック", interactive=False)
            

        gr.Markdown("### ナビゲーション履歴")
        history_display = gr.Textbox(label="履歴", lines=20, interactive=False)

    outputs_list = [
        current_prompt_display, copyable_prompt_textbox, negative_prompt_textbox, feedback_text,
        structured_history_log, history_display, current_keywords_state, 
        current_negative_keywords_state, feedback_input
    ]

    generation_outputs = [
        feedback_text, generated_image, generated_image_state, extracted_tags_label, 
        tags_state, selected_tags
    ]

    tagger_outputs = [
        feedback_text, extracted_tags_label, tags_state, selected_tags
    ]

    start_button.click(fn=start_navigation, inputs=[initial_prompt_input], outputs=outputs_list)
    refine_button.click(fn=handle_refinement, inputs=[feedback_input, structured_history_log, current_keywords_state, current_negative_keywords_state, copyable_prompt_textbox, negative_prompt_textbox], outputs=outputs_list)
    generate_button.click(fn=handle_image_generation, inputs=[copyable_prompt_textbox, negative_prompt_textbox], outputs=generation_outputs)
    tagger_button.click(fn=handle_tag_extraction, inputs=[generated_image_state], outputs=tagger_outputs)
    
    # 選択したタグを追加するボタンのイベント
    add_tags_button.click(
        fn=add_selected_tags_to_prompt, 
        inputs=[selected_tags, copyable_prompt_textbox], 
        outputs=[feedback_text, copyable_prompt_textbox]
    )
    
    # 選択したタグをネガティブプロンプトに追加するボタンのイベント
    add_negative_tags_button.click(
        fn=add_selected_tags_to_negative_prompt, 
        inputs=[selected_tags, negative_prompt_textbox], 
        outputs=[feedback_text, negative_prompt_textbox]
    )

# --- 6. アプリケーションの起動 ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",server_port=7878, debug=True)
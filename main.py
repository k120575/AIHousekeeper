import os
import asyncio
import logging
import threading
from dotenv import load_dotenv
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from google import genai
from supabase import create_client, Client
import httpx
import re

# 1. 初始化環境與日誌
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 2. 從環境變數讀取
TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")
S_URL = os.getenv("SUPABASE_URL")
S_KEY = os.getenv("SUPABASE_KEY")

if not all([TOKEN, API_KEY, S_URL, S_KEY]):
    print("❌ 錯誤：.env 檔案內容不完整。")
    exit(1)

# 3. 初始化全局 Client
client = genai.Client(api_key=API_KEY)
supabase: Client = create_client(S_URL, S_KEY)

# ================= 核心工具函數 (保持原樣) =================

async def get_or_create_user(user_id: int):
    try:
        res = supabase.table("user_profile").select("*").eq("user_id", user_id).execute()
        if not res.data:
            data = supabase.table("user_profile").insert({"user_id": user_id}).execute()
            return data.data[0]
        return res.data[0]
    except Exception as e:
        logger.error(f"Supabase Profile Error: {e}")
        return {"personality_summary": "觀察中", "user_id": user_id}

async def get_semantic_memories(user_id: int, text: str):
    try:
        emb = client.models.embed_content(model="text-embedding-004", contents=text)
        vector = emb.embeddings[0].values

        rpc_res = supabase.rpc("match_memories", {
            "query_embedding": vector,
            "match_threshold": 0.4,
            "match_count": 3,
            "p_user_id": user_id
        }).execute()
        return "\n".join([r['content'] for r in rpc_res.data]) if rpc_res.data else "尚無相關回憶。"
    except Exception as e:
        logger.error(f"Memory Search Error: {e}")
        return ""

def get_weather_context(text: str) -> str:
    """如果對話包含天氣關鍵字，查詢天氣。"""
    if "天氣" not in text and "weather" not in text.lower():
        return ""
    
    # 簡單的地點提取 (預設台北)
    city = "Taipei"
    # 常見台灣城市映射
    city_map = {
        "台北": "Taipei", "臺北": "Taipei",
        "台中": "Taichung", "臺中": "Taichung",
        "高雄": "Kaohsiung", "台南": "Tainan", "新竹": "Hsinchu",
        "桃園": "Taoyuan"
    }
    for k, v in city_map.items():
        if k in text:
            city = v
            break
            
    try:
        # j1 format returns JSON
        url = f"https://wttr.in/{city}?format=j1"
        res = httpx.get(url, timeout=3.0)
        if res.status_code == 200:
            data = res.json()
            curr = data['current_condition'][0]
            desc = curr['lang_zh-TV'][0]['value'] if 'lang_zh-TV' in curr else curr['weatherDesc'][0]['value']
            temp = curr['temp_C']
            feels = curr['FeelsLikeC']
            return f"\n[即時天氣數據 - {city}] 狀態：{desc}，氣溫：{temp}°C (體感 {feels}°C)。(請依據此數據回答，不可瞎掰)"
    except Exception as e:
        logger.error(f"Weather API Error: {e}")
    
    return ""

# ================= 訊息處理流程 (補上 chat_logs 邏輯) =================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return

    user_id = update.effective_user.id
    user_input = update.message.text

    try:
        profile = await get_or_create_user(user_id)
        past_memories = await get_semantic_memories(user_id, user_input)
        weather_info = get_weather_context(user_input)

        # --- 優化後的 System Prompt (極簡版) ---
        system_prompt = f"""
        # Role
        你是一位極度精簡、專業的一流管家。

        # 核心指令
        1. **極度精簡**：回答必須少於 50 字，除非必要。直接講重點，省去所有客套（如「好的」、「明白」）。
        2. **實事求是**：參考提供的[即時天氣數據]回答天氣，不可瞎掰。
        3. **稱呼**：自然地使用閣下的稱呼（{profile['personality_summary'] or '閣下'}）。

        # 資訊
        - 記憶：{past_memories}
        {weather_info}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n\n閣下現在說：{user_input}"
        )

        bot_reply = response.text
        await update.message.reply_text(bot_reply)

        asyncio.create_task(background_evolution(user_id, user_input, profile['personality_summary'], bot_reply))

    except Exception as e:
        logger.error(f"Main Loop Error: {e}")
        await update.message.reply_text("抱歉，閣下。我的思緒稍微紊亂了，請容我重新整理。")

async def background_evolution(user_id, text, old_summary, bot_reply):
    """背景執行：存入 Log + 存入記憶 + 性格進化"""
    try:
        # A. 存入原始對話 chat_logs
        supabase.table("chat_logs").insert({
            "user_id": user_id,
            "user_text": text,
            "bot_text": bot_reply
        }).execute()

        # B. 存入長期記憶 (向量化)
        emb = client.models.embed_content(model="text-embedding-004", contents=text)
        supabase.table("long_term_memories").insert({
            "user_id": user_id,
            "content": text,
            "embedding": emb.embeddings[0].values
        }).execute()

        # C. 性格演化
        reflect_prompt = f"分析此對話並更新描述：{text}。目前認知：{old_summary}"
        try:
            res = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=reflect_prompt
            )
            supabase.table("user_profile").update({
                "personality_summary": res.text
            }).eq("user_id", user_id).execute()
        except Exception as api_err:
            logger.warning(f"Evolution task paused (Quota?): {api_err}")

    except Exception as e:
        logger.error(f"Background Task Error: {e}")

# ================= 啟動執行 (統一合併 Flask 與 Bot) =================

server = Flask(__name__)

@server.route('/')
def home():
    return "I'm alive!"

def run_web():
    port = int(os.environ.get("PORT", 8080))
    server.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    # 1. 初始化 Bot
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # 2. 檢查模式 (Webhook vs Polling)
    # Render 等平台會自動提供 RENDER_EXTERNAL_URL 或我們自己設定 WEBHOOK_URL
    WEBHOOK_URL = os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL")
    PORT = int(os.environ.get("PORT", 8080))

    if WEBHOOK_URL:
        # --- Webhook 模式 (雲端部署用) ---
        print(f"--- 🚀 啟動 Webhook 模式 (Port {PORT}) ---")
        print(f"--- URL: {WEBHOOK_URL} ---")
        
        # 啟動 Webhook，同時監聽 Port，這樣就不需要額外的 Flask Server 了
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path="telegram",
            webhook_url=f"{WEBHOOK_URL}/telegram",
            drop_pending_updates=True
        )
    else:
        # --- Polling 模式 (本地開發用) ---
        print("--- 🐢 啟動 Polling 模式 (本地開發) ---")
        print("--- 正在啟動 Flask 健康檢查伺服器 (保持相容性) ---")
        # 只有在 Polling 模式才需要額外開 Flask 來佔用 Port (如果平台強制要求)
        threading.Thread(target=run_web, daemon=True).start()
        
        print("--- ✅ 管家啟動成功 ---")
        app.run_polling(drop_pending_updates=True)
import os
import asyncio
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
# 確保這是你環境中安裝的新版 google-genai
from google import genai
from supabase import create_client, Client

# 1. 初始化環境與日誌
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 屏蔽掉討厭的連線 Log
logging.getLogger("httpx").setLevel(logging.WARNING)

# 2. 從環境變數讀取 (檢查點)
TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")
S_URL = os.getenv("SUPABASE_URL")
S_KEY = os.getenv("SUPABASE_KEY")

if not all([TOKEN, API_KEY, S_URL, S_KEY]):
    print("❌ 錯誤：.env 檔案內容不完整，請檢查 Key 名稱是否正確。")
    exit(1)

# 3. 初始化全局 Client
client = genai.Client(api_key=API_KEY)
supabase: Client = create_client(S_URL, S_KEY)

# ================= 核心工具函數 =================

async def get_or_create_user(user_id: int):
    """資料庫操作：確保用戶存在"""
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
    """向量搜尋記憶"""
    try:
        # 新版 SDK 語法：embeddings[0].values
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

# ================= 訊息處理流程 =================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return

    user_id = update.effective_user.id
    user_input = update.message.text

    try:
        # 1. 取得背景資料 (並行處理以加速)
        profile = await get_or_create_user(user_id)
        past_memories = await get_semantic_memories(user_id, user_input)

        # 2. 準備 Prompt (使用配額穩定的 2.5-flash)
        system_prompt = f"你是一位專業管家。當前認知：{profile['personality_summary']}\n記憶：{past_memories}"

        # 3. AI 回覆
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n\n主人說：{user_input}"
        )

        await update.message.reply_text(response.text)

        # 4. 觸發背景任務 (儲存與進化)
        asyncio.create_task(background_evolution(user_id, user_input, profile['personality_summary']))

    except Exception as e:
        logger.error(f"Main Loop Error: {e}")
        await update.message.reply_text("抱歉，我現在有點短路，請稍後再試。")

async def background_evolution(user_id, text, old_summary):
    """背景執行：存入記憶 + 性格進化"""
    try:
        # A. 存入長期記憶
        emb = client.models.embed_content(model="text-embedding-004", contents=text)
        supabase.table("long_term_memories").insert({
            "user_id": user_id,
            "content": text,
            "embedding": emb.embeddings[0].values
        }).execute()

        # B. 性格演化 (只有在背景默默做，掛了也不影響對話)
        reflect_prompt = f"分析此對話並更新描述：{text}。目前認知：{old_summary}"

        # 嘗試使用 3-Flash (若 429 報錯則不更新)
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

# ================= 啟動執行 =================

import threading
from flask import Flask

# 1. 建立 Flask Server (放在外面或裡面皆可，這裡放外層較清晰)
server = Flask(__name__)

@server.route('/')
def home():
    return "I'm alive!"

def run_web():
    # 這裡必須抓取 Render 提供的 PORT 變數
    port = int(os.environ.get("PORT", 8080))
    # host 必須是 0.0.0.0 才能讓外部掃描到
    server.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    # A. 啟動 Web 服務執行緒 (daemon=True 表示主程式結束時它也會跟著結束)
    print("--- 正在啟動 Flask 健康檢查伺服器 ---")
    threading.Thread(target=run_web, daemon=True).start()

    # B. 初始化 Telegram Bot
    print("--- ✅ 管家正在啟動 ---")
    app = ApplicationBuilder().token(TOKEN).build()

    # 註冊處理器
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    # C. 開始輪詢 (這行會阻塞主執行緒，所以必須放在最後)
    app.run_polling(drop_pending_updates=True)
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

# ================= 訊息處理流程 (補上 chat_log 邏輯) =================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return

    user_id = update.effective_user.id
    user_input = update.message.text

    try:
        profile = await get_or_create_user(user_id)
        past_memories = await get_semantic_memories(user_id, user_input)

        # --- 優化後的 System Prompt ---
        system_prompt = f"""
        # Role
        你是一位觀察入微、優雅且專業的私人家臣管家。你的目標是根據閣下的性格與過去偏好，提供高度客製化的情感價值與生活建議。

        # 閣下檔案 (核心認知)
        - 性格總結：{profile['personality_summary'] if profile['personality_summary'] else '初次見面，正在觀察中...'}
        - 關鍵記憶片段：{past_memories}

        # 互動準則
        1. **稱呼與記憶**：請務必從「性格總結」或「記憶片段」中尋找閣下的姓名或慣用稱呼。如果知道閣下是誰，請在適當時機自然地稱呼他。
        2. **語氣**：保持謙遜但有見地的態度。避免機器人的刻板回覆，多一點人性化的觀察，就像19世紀英國貴族的管家，例如黑執事。
        3. **連續性**：如果閣下提到的內容與過去記憶相關，請主動連結，例如：「閣下，這跟您上次提到的...似乎有關？」
        4. **進化**：你的一言一行都在形塑閣下的生活，請保持對細節的敏銳度。
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
        # A. 存入原始對話 chat_log
        supabase.table("chat_log").insert({
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
    # 1. 啟動 Web 服務執行緒 (解決 Render Port Scan 問題)
    print("--- 正在啟動 Flask 健康檢查伺服器 ---")
    threading.Thread(target=run_web, daemon=True).start()

    # 2. 啟動 Telegram Bot
    print("--- ✅ 管家啟動成功 ---")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    app.run_polling(drop_pending_updates=True)
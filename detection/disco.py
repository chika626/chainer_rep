# インストールした discord.py を読み込む
import discord
import asyncio
import sys
import urllib.request
import inference
import io
import random

# 自分のBotのアクセストークンに置き換えてください
TOKEN = 'Nzg1NTIyMDQ1Mzg3NTM4NDgz.X85EaQ.jX3Pwc_kHLApM-U-7Pgtg6vbJO4'

# 接続に必要なオブジェクトを生成
client = discord.Client()

# 起動時に動作する処理
@client.event
async def on_ready():
    # 起動したらターミナルにログイン通知が表示される
    print('ログインしました')
    channel = client.get_channel(785523541281079328)
    await channel.send('hello')

# メッセージ受信時に動作する処理
@client.event
async def on_message(message):
    # メッセージ送信者がBotだった場合は無視する
    if message.author.bot:
        return

    # channnelを指定
    channel = client.get_channel(785523541281079328)

    # 「/neko」と発言したら「にゃーん」が返る処理
    if message.content == '/neko':
        await message.channel.send('にゃーん')
    elif not message.attachments:
        await channel.send("no attachments")
    else:
        # 中身の確認
        # await channel.send(message.attachments[0].filename)

        await channel.send("ok. start inference.")

        # 403回避
        url = message.attachments[0].url
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0",
        }
        request = urllib.request.Request(url=url, headers=headers)
        savename = "test.png"
        file = io.BytesIO(urllib.request.urlopen(request).read())
        inference.discord_inf(file)
        filepath = 'fin_inf.jpg'
        await channel.send('推論結果',file=discord.File(filepath))

    

# @client.event
# async def on_message(message): 
#     if message.content.startswith('/exit'):#!SHUTDOWN_BOTが入力されたら強制終了
#         await client.logout()
#         await sys.exit()

# Botの起動とDiscordサーバーへの接続
client.run(TOKEN)
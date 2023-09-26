import asyncio
import edge_tts

Text=''
with open('C:\\users\\22638\\Desktop\\tts.txt','rb') as f:
    data=f.read()
    Text=data.decode('utf-8')
print(Text)
voice1='zh-CN-XiaoyiNeural'
voice2='zh-CN-XiaoxiaoNeural'
voice3='en-US-AnaNeural'
voice4='zh-CN-YunjianNeural'
voice5='ja-JP-NanamiNeural'
output='C:\\users\\22638\\Desktop\\tts.mp3'
rate='-15%'
volume='+50%'

async def my_function():
        tts=edge_tts.Communicate(text=Text,voice=voice2,rate=rate,volume=volume)
        await tts.save(output)

if __name__ == '__main__':
    asyncio.run(my_function())

import yt_dlp

url = "https://www.youtube.com/watch?v=OZzfotR8chQ&list=RDOZzfotR8chQ&start_radio=1"

ydl_opts = {
    'outtmpl': '%(title)s.%(ext)s',  # Save as video title
    'format': 'best',                 # Download best quality video
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

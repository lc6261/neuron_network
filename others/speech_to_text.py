import whisper
import os
import torch
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel

# 指定模型路径
model_path = r"C:\Users\lc626\.cache\lm-studio\models\vonjack\whisper-large-v3-gguf\whisper-large-v3-q8_0.gguf"

# 明确指定 FFmpeg 路径
ffmpeg_path = r"E:\tools\ffmpeg-master-latest-win64-gpl\bin"  # 请确保这是正确的路径
os.environ["PATH"] += os.pathsep + ffmpeg_path

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    video.close()

def transcribe_with_whisper(file_path, language="zh"):
    # 检查文件类型
    _, file_extension = os.path.splitext(file_path)
    
    # 如果是视频文件,先提取音频
    if file_extension.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        audio_file_path = "temp_audio.wav"
        extract_audio(file_path, audio_file_path)
    else:
        audio_file_path = file_path

    # 检查是否有可用的 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载模型
    # 可选的模型类型包括:
    # "tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"
    model = whisper.load_model("medium", device=device)

    # 执行转录,指定语言为中文,并启用标点符号
    result = model.transcribe(audio_file_path, language=language, task="transcribe", fp16=False)

    # 后处理：添加段落
    processed_text = ""
    sentence_count = 0
    for segment in result["segments"]:
        processed_text += segment["text"].strip() + " "
        sentence_count += 1
        if sentence_count >= 3 and segment["text"].strip().endswith(("。", "！", "？")):
            processed_text += "\n\n"
            sentence_count = 0

    # 如果创建了临时音频文件,删除它
    if audio_file_path != file_path:
        os.remove(audio_file_path)

    return processed_text

def transcribe_with_faster_whisper(file_path, language="zh"):
    # 检查文件类型
    _, file_extension = os.path.splitext(file_path)
    
    # 如果是视频文件,先提取音频
    if file_extension.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        audio_file_path = "temp_audio.wav"
        extract_audio(file_path, audio_file_path)
    else:
        audio_file_path = file_path

    # 加载模型，执行转录,指定语言为中文
    model = WhisperModel(model_path, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio_file_path, language=language)
    
    # 收集转录结果
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    # 如果创建了临时音频文件,删除它
    if audio_file_path != file_path:
        os.remove(audio_file_path)

    return transcribed_text

# 使用示例
file_path = r"H:\bilibili\Python手搓神经网络\1.12. 反向传播之需求函数 - Python手搓神经网络(Av491866276,P1).mp4"

# 使用原始Whisper模型转录
whisper_result = transcribe_with_whisper(file_path, language="zh")
print("Whisper转录结果:")
print(whisper_result)

# 将转录结果保存到文本文件
with open("whisper_result.txt", "w", encoding="utf-8") as f:
    f.write(whisper_result)

print("转录结果已保存到 whisper_result.txt")


'''
# 使用Faster Whisper模型转录
faster_whisper_result = transcribe_with_faster_whisper(file_path, language="zh")
print("Faster Whisper转录结果:")
print(faster_whisper_result)

with open("faster_whisper_result.txt", "w", encoding="utf-8") as f:
    f.write(faster_whisper_result)

print("转录结果已分别保存到 whisper_result.txt 和 faster_whisper_result.txt")
'''

import whisper
import torch
from pyannote.audio import Pipeline
from moviepy.editor import VideoFileClip
import os

class AudioProcessor:
    def __init__(self, whisper_model="medium", language="zh"):
        self.whisper_model = whisper.load_model(whisper_model)
        self.language = language
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_HF_TOKEN")
        
        # 检查是否有可用的 GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

    def extract_audio(self, video_path, audio_path):
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
        video.close()

    def transcribe_with_diarization(self, file_path):
        # 检查文件类型
        _, file_extension = os.path.splitext(file_path)
        
        # 如果是视频文件,先提取音频
        if file_extension.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            audio_file_path = "temp_audio.wav"
            self.extract_audio(file_path, audio_file_path)
        else:
            audio_file_path = file_path

        # 使用 Whisper 进行语音转文本
        result = self.whisper_model.transcribe(audio_file_path, language=self.language)

        # 使用 pyannote 进行说话人分割
        diarization = self.diarization_pipeline(audio_file_path)

        # 合并说话人分割和转文本结果
        final_output = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            relevant_text = [segment for segment in result["segments"] 
                             if segment["start"] >= turn.start and segment["end"] <= turn.end]
            text = " ".join([segment["text"] for segment in relevant_text])
            if text.strip():  # 只有在有文本时才添加
                final_output.append(f"说话人 {speaker}: {text}")

        # 如果创建了临时音频文件,删除它
        if audio_file_path != file_path:
            os.remove(audio_file_path)

        return "\n".join(final_output)

    def process_file(self, file_path):
        try:
            result = self.transcribe_with_diarization(file_path)
            print("处理完成。结果如下：")
            print(result)
            
            # 将结果保存到文件
            output_file = "transcription_with_speakers.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"结果已保存到 {output_file}")
            
        except Exception as e:
            print(f"处理文件时发生错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    processor = AudioProcessor()
    file_path = r"H:\bilibili\Python手搓神经网络\1.12. 反向传播之需求函数 - Python手搓神经网络(Av491866276,P1).mp4"
    processor.process_file(file_path)

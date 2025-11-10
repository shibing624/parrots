# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: FastAPI client for IndexTTS2 TTS generation with audio playback
"""
import os
import sys
import time
import requests
import tempfile
from pathlib import Path
from typing import Optional
import sounddevice as sd
import librosa


class TTSClient:
    """IndexTTS2 FastAPI客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: FastAPI服务器地址
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.temp_dir = Path(tempfile.gettempdir()) / "indextts_client_audio"
        self.temp_dir.mkdir(exist_ok=True)
    
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                print("✓ Server is healthy")
                return True
            else:
                print(f"✗ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to server: {e}")
            return False
    
    def download_audio(self, file_id: str, save_path: str) -> bool:
        """
        从服务器下载音频文件
        
        Args:
            file_id: 音频文件ID
            save_path: 保存路径
        
        Returns:
            是否下载成功
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/audio/{file_id}",
                timeout=30
            )
            response.raise_for_status()
            
            # 保存文件
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"✗ Failed to download audio: {e}")
            return False
    
    def generate_tts(
        self,
        text: str,
        speak_reference_audio_path_or_name: str = "male_broadcaster",
        emo_reference_audio_path: Optional[str] = None,
        emo_alpha: float = 1.0,
        emo_vector: Optional[list] = None,
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        use_random: bool = False,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        verbose: bool = False,
        save_path: Optional[str] = None,
        play_audio: bool = True
    ) -> Optional[str]:
        """
        生成TTS音频
        
        Args:
            text: 要合成的文本
            speak_reference_audio_path_or_name: 参考音频路径或内置音色名称
            emo_reference_audio_path: 情感参考音频路径
            emo_alpha: 情感混合因子
            emo_vector: 情感向量
            use_emo_text: 是否使用文本作为情感参考
            emo_text: 用于生成情感向量的文本
            use_random: 是否使用随机情感向量
            interval_silence: 生成片段之间的静音间隔（毫秒）
            max_text_tokens_per_segment: 每个片段的最大文本token数
            verbose: 是否输出详细日志
            save_path: 保存音频的路径，如果为None则使用临时文件
            play_audio: 是否播放音频
        
        Returns:
            音频文件路径，如果失败返回None
        """
        # 构建请求数据
        request_data = {
            "text": text,
            "speak_reference_audio_path_or_name": speak_reference_audio_path_or_name,
            "emo_alpha": emo_alpha,
            "use_emo_text": use_emo_text,
            "use_random": use_random,
            "interval_silence": interval_silence,
            "max_text_tokens_per_segment": max_text_tokens_per_segment,
            "verbose": verbose
        }
        
        if emo_reference_audio_path:
            request_data["emo_reference_audio_path"] = emo_reference_audio_path
        if emo_vector:
            request_data["emo_vector"] = emo_vector
        if emo_text:
            request_data["emo_text"] = emo_text
        
        # 发送请求
        print(f"\n📝 Generating TTS for text: {text[:50]}...")
        try:
            response = self.session.post(
                f"{self.base_url}/api/tts",
                json=request_data,
                timeout=300  # 5分钟超时
            )
            response.raise_for_status()
            result = response.json()
            
            if not result.get("success"):
                print(f"✗ TTS generation failed: {result.get('message')}")
                return None
            
            # 获取服务器返回的文件ID
            file_id = result.get("file_id")
            if not file_id:
                print("✗ No file_id in response")
                return None
            
            print(f"✓ Audio generated on server with file_id: {file_id}")
            
            # 确定本地保存路径
            if save_path is None:
                save_path = str(self.temp_dir / f"{file_id}.wav")
            else:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            # 从服务器下载音频文件
            print(f"📥 Downloading audio to: {save_path}")
            if not self.download_audio(file_id, save_path):
                return None
            
            print(f"✓ Audio downloaded successfully")
            
            # 播放音频
            if play_audio:
                self.play_audio(save_path)

            return save_path
        
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def play_audio(self, audio_path: str, sample_rate: int = 22050):
        """
        播放音频文件
        
        Args:
            audio_path: 音频文件路径
            sample_rate: 采样率
        """
        try:
            # 使用librosa加载音频（自动处理采样率）
            audio, sr = librosa.load(audio_path, sr=None)
            print(f"🔊 Playing audio (duration: {len(audio)/sr:.2f}s, sample_rate: {sr}Hz)...")
            sd.play(audio, sr)
            sd.wait()  # 等待播放完成
        except Exception as e:
            print(f"⚠️  Error playing audio: {e}")


def demo_basic_tts(server_url: str = "http://localhost:8000"):
    """基础TTS演示"""
    print("\n" + "="*60)
    print("Demo 1: Basic TTS Generation")
    print("="*60)
    
    client = TTSClient(base_url=server_url)
    if not client.check_server_health():
        return

    text = "欢迎大家来体验IndexTTS2，这是一个高质量的文本转语音系统。"
    audio_path = client.generate_tts(
        text=text,
        speak_reference_audio_path_or_name="male_broadcaster",
        play_audio=True
    )
    
    if audio_path:
        print(f"✓ Demo 1 completed. Audio saved at: {audio_path}")


def demo_emotion_tts(server_url: str = "http://localhost:8000"):
    """情感TTS演示"""
    print("\n" + "="*60)
    print("Demo 2: Emotion-based TTS Generation")
    print("="*60)
    
    client = TTSClient(base_url=server_url)
    if not client.check_server_health():
        return
    
    # 使用文本情感分析
    text = "快躲起来！是他要来了！他要来抓我们了！"
    audio_path = client.generate_tts(
        text=text,
        speak_reference_audio_path_or_name="male_broadcaster",
        use_emo_text=True,
        emo_alpha=0.6,
        play_audio=True
    )
    
    if audio_path:
        print(f"✓ Demo 2 completed. Audio saved at: {audio_path}")


def demo_custom_emotion_vector(server_url: str = "http://localhost:8000"):
    """自定义情感向量演示"""
    print("\n" + "="*60)
    print("Demo 3: Custom Emotion Vector TTS")
    print("="*60)
    
    client = TTSClient(base_url=server_url)
    if not client.check_server_health():
        return
    
    # 自定义情感向量: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    # 设置高兴情绪
    emo_vector = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0]
    
    text = "今天天气真好，我们一起去公园玩吧！"
    audio_path = client.generate_tts(
        text=text,
        speak_reference_audio_path_or_name="female_young",
        emo_vector=emo_vector,
        emo_alpha=1.0,
        play_audio=True
    )
    
    if audio_path:
        print(f"✓ Demo 3 completed. Audio saved at: {audio_path}")


def demo_different_voices(server_url: str = "http://localhost:8000"):
    """不同音色演示"""
    print("\n" + "="*60)
    print("Demo 4: Different Voice Styles")
    print("="*60)
    
    client = TTSClient(base_url=server_url)
    if not client.check_server_health():
        return
    
    text = "这是一个测试不同音色的演示。"
    voices = [
        ("male_broadcaster", "中文男主播"),
        ("female_young", "傲娇御姐"),
        ("male_mature", "沉稳高管"),
    ]
    
    for voice_name, voice_desc in voices:
        print(f"\n🎤 Using voice: {voice_desc} ({voice_name})")
        audio_path = client.generate_tts(
            text=text,
            speak_reference_audio_path_or_name=voice_name,
            play_audio=True
        )
        if audio_path:
            time.sleep(1)  # 短暂停顿


def demo_long_text(server_url: str = "http://localhost:8000"):
    """长文本演示"""
    print("\n" + "="*60)
    print("Demo 5: Long Text TTS")
    print("="*60)
    
    client = TTSClient(base_url=server_url)
    if not client.check_server_health():
        return
    
    long_text = """
    人工智能技术的发展正在改变我们的生活方式。从语音助手到自动驾驶汽车，
    从医疗诊断到金融分析，AI技术正在各个领域发挥重要作用。
    文本转语音技术作为AI的一个重要分支，为视障人士提供了便利，
    也为内容创作者提供了新的工具。IndexTTS2是一个高质量的文本转语音系统，
    能够生成自然流畅的语音，支持多种音色和情感表达。
    """
    
    audio_path = client.generate_tts(
        text=long_text.strip(),
        speak_reference_audio_path_or_name="male_broadcaster",
        max_text_tokens_per_segment=100,
        play_audio=True
    )
    
    if audio_path:
        print(f"✓ Demo 5 completed. Audio saved at: {audio_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IndexTTS2 FastAPI Client Demo")
    parser.add_argument("--server", type=str, default="http://localhost:8005", help="Server URL")
    parser.add_argument("--demo", type=int, choices=[1, 2, 3, 4, 5], help="Run specific demo (1-5)", default=1)
    parser.add_argument("--all", action="store_true", help="Run all demos")
    args = parser.parse_args()
    
    # 全局服务器地址，供demo函数使用
    global_server_url = args.server
    
    if args.all:
        # 运行所有演示
        demo_basic_tts(server_url=global_server_url)
        time.sleep(2)
        demo_emotion_tts(server_url=global_server_url)
        time.sleep(2)
        demo_custom_emotion_vector(server_url=global_server_url)
        time.sleep(2)
        demo_different_voices(server_url=global_server_url)
        time.sleep(2)
        demo_long_text(server_url=global_server_url)
    elif args.demo:
        # 运行指定演示
        demos = {
            1: demo_basic_tts,
            2: demo_emotion_tts,
            3: demo_custom_emotion_vector,
            4: demo_different_voices,
            5: demo_long_text
        }
        demos[args.demo](server_url=global_server_url)
    else:
        # 默认运行基础演示
        print("Running default demo. Use --all to run all demos or --demo N for specific demo.")
        demo_basic_tts(server_url=global_server_url)


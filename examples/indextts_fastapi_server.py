# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: FastAPI server for IndexTTS2 TTS generation
"""
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.append('..')
from parrots.indextts.inference import IndexTTS2
from parrots.log import set_log_level, logger

# 设置日志级别
set_log_level("DEBUG")

# 初始化FastAPI应用
app = FastAPI(
    title="IndexTTS2 TTS API",
    description="Text-to-Speech API using IndexTTS2",
    version="1.0.0"
)

# 创建API路由器，添加/api前缀
from fastapi import APIRouter
api_router = APIRouter(prefix="/api")

# 全局TTS模型实例
tts_model: Optional[IndexTTS2] = None


class TTSRequest(BaseModel):
    """TTS请求模型"""
    text: str = Field(..., description="要合成的文本")
    speak_reference_audio_path_or_name: str = Field(
        default="male_broadcaster",
        description="参考音频路径或内置音色名称: 'male_broadcaster', 'female_young', 'male_mature', 'male_magnetic', 'male_cantonese'"
    )
    emo_reference_audio_path: Optional[str] = Field(
        default=None,
        description="情感参考音频路径，默认使用说话人参考音频"
    )
    emo_alpha: float = Field(
        default=1.0,
        description="情感混合因子，范围0.0-1.0",
        ge=0.0,
        le=1.0
    )
    emo_vector: Optional[List[float]] = Field(
        default=None,
        description="情感向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]"
    )
    use_emo_text: bool = Field(
        default=False,
        description="是否使用文本作为情感参考"
    )
    emo_text: Optional[str] = Field(
        default=None,
        description="用于生成情感向量的文本，默认使用主文本"
    )
    use_random: bool = Field(
        default=False,
        description="是否使用随机情感向量"
    )
    interval_silence: int = Field(
        default=200,
        description="生成片段之间的静音间隔（毫秒）",
        ge=0
    )
    max_text_tokens_per_segment: int = Field(
        default=120,
        description="每个片段的最大文本token数",
        ge=1
    )
    verbose: bool = Field(
        default=True,
        description="是否输出详细日志"
    )


class TTSResponse(BaseModel):
    """TTS响应模型"""
    success: bool
    message: str
    audio_path: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化TTS模型"""
    global tts_model
    try:
        logger.info("Initializing IndexTTS2 model...")
        # 从app.state获取model_dir参数
        model_dir = getattr(app.state, 'model_dir', None)
        if model_dir:
            logger.info(f"Using model directory: {model_dir}")
            tts_model = IndexTTS2(model_dir=model_dir)
        else:
            tts_model = IndexTTS2()
        logger.info("IndexTTS2 model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize IndexTTS2 model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    global tts_model
    if tts_model is not None:
        logger.info("Shutting down IndexTTS2 model...")
        tts_model = None


@api_router.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "IndexTTS2 TTS API",
        "version": "1.0.0",
        "endpoints": {
            "/api/tts": "POST - Generate TTS audio",
            "/api/health": "GET - Health check"
        }
    }


@api_router.get("/health")
async def health_check():
    """健康检查"""
    if tts_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "TTS model not initialized"}
        )
    return {"status": "healthy", "message": "TTS model is ready"}


# 临时目录用于存储生成的音频文件
temp_dir = Path(tempfile.gettempdir()) / "indextts_audio"
temp_dir.mkdir(exist_ok=True)


@api_router.post("/tts", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """
    生成TTS音频
    
    返回生成的音频文件路径，客户端可以通过 /tts/audio/{file_id} 下载
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not initialized")
    
    try:
        # 生成唯一的文件ID
        file_id = f"tts_{int(time.time() * 1000)}_{hash(request.text) % 10000}"
        output_path = str(temp_dir / f"{file_id}.wav")
        
        # 调用TTS生成
        logger.info(f"Generating TTS for text: {request.text[:50]}...")
        result = tts_model.infer(
            text=request.text,
            speak_reference_audio_path_or_name=request.speak_reference_audio_path_or_name,
            output_path=output_path,
            emo_reference_audio_path=request.emo_reference_audio_path,
            emo_alpha=request.emo_alpha,
            emo_vector=request.emo_vector,
            use_emo_text=request.use_emo_text,
            emo_text=request.emo_text,
            use_random=request.use_random,
            interval_silence=request.interval_silence,
            verbose=request.verbose,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        
        logger.info(f"TTS generated successfully: {output_path}")
        
        return TTSResponse(
            success=True,
            message="TTS generated successfully",
            audio_path=output_path
        )
    
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")


# 将API路由器包含到主应用中
app.include_router(api_router)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IndexTTS2 FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8005, help="Port to bind")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to model directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # 将model_dir保存到app.state中，供startup_event使用
    app.state.model_dir = args.model_dir
    
    logger.info(f"Starting IndexTTS2 FastAPI server on {args.host}:{args.port}")
    if args.model_dir:
        logger.info(f"Model directory: {args.model_dir}")
    uvicorn.run(
        "indextts_fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from litellm import completion
from models.chat_model import ChatRequest
from services.chat_service import ChatService
import json
import os
import traceback
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'xiaozhi-backend-FastAPI running ok'}

@app.post('/v1/chat/completions')
async def chat_completions(request: ChatRequest):
    """
    实现OpenAI兼容的聊天完成接口
    """
    try:
        if request.stream:
            # 流式响应
            async def generate_stream():
                try:
                    async for chunk in ChatService.chat_completion(request):
                        # 根据OpenAI规范格式化响应
                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                    
                    # 结束标记
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    # 流式传输中的错误处理
                    error_data = {
                        "error": {
                            "message": f"Stream error: {str(e)}",
                            "type": "stream_error",
                            "stack": traceback.format_exc()
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # 非流式响应 - 收集所有数据然后返回
            full_response = ""
            session_id = ""
            model_name = ""
            created_time = 0
            finish_reason = "stop"
            
            try:
                async for chunk in ChatService.chat_completion(request):
                    if chunk.choices and chunk.choices[0].delta.get("content"):
                        full_response += chunk.choices[0].delta["content"]
                        if not session_id:
                            session_id = chunk.id
                            model_name = chunk.model
                            created_time = chunk.created
                            finish_reason = chunk.choices[0].finish_reason or "stop"
                            
                # 返回完整响应（简化版）
                return {
                    "id": session_id,
                    "object": "chat.completion",
                    "created": created_time,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_response
                        },
                        "finish_reason": finish_reason
                    }]
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": {
                            "message": f"Completion error: {str(e)}",
                            "type": "completion_error",
                            "stack": traceback.format_exc()
                        }
                    }
                )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Server error: {str(e)}",
                    "type": "server_error",
                    "stack": traceback.format_exc()
                }
            }
        )
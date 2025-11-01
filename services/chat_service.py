import litellm
import asyncio
import uuid
import time
from typing import AsyncGenerator, Optional
from models.chat_model import ChatRequest, ChatResponse, ChatChoice
import os
from openai import OpenAI, AsyncOpenAI
import httpx


class ChatService:
    @staticmethod
    async def chat_completion(request: ChatRequest) -> AsyncGenerator[ChatResponse, None]:
        """
        使用LiteLLM进行聊天完成
        
        Args:
            request: 聊天请求
            
        Yields:
            ChatResponse: 流式的聊天响应
        """
        # 检查是否为阿里云百炼平台的模型
        if request.model and "qwen" in request.model.lower():
            async for chunk in ChatService._aliyun_bailian_chat_completion(request):
                yield chunk
            return
        
        # 添加必要的参数
        kwargs = {
            "model": request.model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "stream": True  # 我们总是使用流式传输
        }
        
        # 处理base_url参数
        if request.base_url:
            kwargs["api_base"] = request.base_url
            
        # 根据模型名称和base_url设置相应的API密钥
        if not request.base_url:  # 如果没有指定base_url，则根据模型名称处理
            if request.model.startswith("deepseek/"):
                kwargs["api_key"] = os.getenv("DEEPSEEK_API_KEY")
            elif request.model.startswith("openai/") or request.model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
                kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
            elif request.model.startswith("qwen/") or "qwen" in request.model.lower():
                kwargs["api_key"] = os.getenv("QWEN_API_KEY")
        else:  # 如果指定了base_url，则根据base_url设置相应的API密钥
            if "deepseek" in request.base_url:
                kwargs["api_key"] = os.getenv("DEEPSEEK_API_KEY")
            elif "openai" in request.base_url:
                kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
            elif "dashscope" in request.base_url or "aliyuncs" in request.base_url:
                kwargs["api_key"] = os.getenv("QWEN_API_KEY")
        
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
            
        # 生成唯一会话ID
        session_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
            
        # 调用LiteLLM
        try:
            response = await asyncio.to_thread(litellm.completion, **kwargs)
        except Exception as e:
            # 错误处理
            error_response = ChatResponse(
                id=session_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        delta={"content": f"Error calling model API: {str(e)}", "role": "assistant"},
                        finish_reason="error"
                    )
                ]
            )
            yield error_response
            return
            
        # 流式传输响应
        try:
            for chunk in response:
                # 如果chunk没有choices，跳过
                if not chunk.choices:
                    continue
                    
                delta_content = chunk.choices[0].delta.content
                
                # 创建响应对象
                chat_response = ChatResponse(
                    id=session_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatChoice(
                            index=0,
                            delta={
                                "content": delta_content if delta_content else "",
                                "role": "assistant" if delta_content else ""
                            },
                            finish_reason=chunk.choices[0].finish_reason
                        )
                    ]
                )
                
                yield chat_response
                
        except Exception as e:
            # 错误处理
            error_response = ChatResponse(
                id=session_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        delta={"content": f"Error during streaming: {str(e)}", "role": "assistant"},
                        finish_reason="error"
                    )
                ]
            )
            yield error_response
    
    @staticmethod
    async def _aliyun_bailian_chat_completion(request: ChatRequest) -> AsyncGenerator[ChatResponse, None]:
        """
        使用阿里云百炼平台进行聊天完成
        
        Args:
            request: 聊天请求
            
        Yields:
            ChatResponse: 流式的聊天响应
        """
        # 生成唯一会话ID
        session_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
        
        # 提取模型名（去除前缀）
        model_name = request.model
        if "/" in request.model:
            model_name = request.model.split("/")[-1]
        
        # 阿里云百炼平台URL
        aliyuncs_base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        
        try:
            # 创建异步OpenAI客户端
            client = AsyncOpenAI(
                api_key=os.getenv("QWEN_API_KEY"),
                base_url=aliyuncs_base_url,
                http_client=httpx.AsyncClient(
                    timeout=60.0,
                ),
            )
            
            # 构造请求参数
            kwargs = {
                "model": model_name,
                "messages": [msg.model_dump() for msg in request.messages],
                "temperature": request.temperature,
                "stream": True
            }
            
            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens
            
            # 调用阿里云百炼平台
            stream = await client.chat.completions.create(**kwargs)
            
            # 流式传输响应
            async for chunk in stream:
                if not chunk.choices:
                    continue
                    
                delta_content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                
                chat_response = ChatResponse(
                    id=chunk.id or session_id,
                    created=chunk.created or created_time,
                    model=chunk.model or request.model,
                    choices=[
                        ChatChoice(
                            index=0,
                            delta={
                                "content": delta_content,
                                "role": "assistant" if delta_content else ""
                            },
                            finish_reason=chunk.choices[0].finish_reason
                        )
                    ]
                )
                
                yield chat_response
                
        except Exception as e:
            # 错误处理
            error_response = ChatResponse(
                id=session_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        delta={"content": f"Error calling Aliyun Bailian API: {str(e)}", "role": "assistant"},
                        finish_reason="error"
                    )
                ]
            )
            yield error_response
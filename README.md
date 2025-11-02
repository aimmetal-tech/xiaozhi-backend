# 小智同学后端服务

- FastAPI 框架

# 演示

当前支持的模型服务商有DeepSeek, OpenAI和阿里云百炼平台

常规
```
Post
{
    "model": "deepseek/deepseek-chat",
    "messages": [
        {
            "role":"user",
            "content": "你好，1+1等于几?"
        }
    ],
    "temperature": 0.7,
    "stream": true
}
```

阿里云百炼
```
Post
{
    
    "model": "qwen/qwen-plus",
    "messages": [
        {
            "role":"user",
            "content": "你好，1+1等于几?"
        }
    ],
    "temperature": 0.7,
    "stream": true
}
```
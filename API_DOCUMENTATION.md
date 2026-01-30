# SAM3 智能标注服务 - 后端API使用手册

## 1. 项目概述

SAM3智能标注服务是一个基于SAM3模型的后端API服务，提供图像和视频的自动标注功能。该服务支持手动提示标注和全自动标注两种模式。

## 2. 服务启动

运行以下命令启动服务：
```bash
python app.py
```

默认监听端口：`8069`
默认主机地址：`0.0.0.0`

## 3. API端点

### 3.1 健康检查
- **端点**: `GET /health`
- **功能**: 检查服务健康状态和模型加载情况
- **返回示例**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3.2 图像标注 - 手动模式
- **端点**: `POST /predict`
- **功能**: 对图像进行标注，支持文本提示和框选提示
- **请求体参数**:
```json
{
  "image_base64": "字符串",      // (可选) Base64编码的图像数据
  "image_path": "字符串",       // (可选) 图像文件路径
  "boxes": [                   // (可选) 框选提示数组
    {
      "box": [x1, y1, x2, y2] // 边界框坐标 [左上角x, 左上角y, 右下角x, 右下角y]
    }
  ],
  "texts": [                   // (可选) 文本提示数组
    {
      "text": "字符串"         // 文本描述
    }
  ]
}
```
- **注意事项**:
  - `image_base64` 和 `image_path` 至少需要提供一个
  - 至少需要提供 `boxes` 或 `texts` 中的一个用于提示
  - `box` 坐标系统为像素坐标系

- **请求示例**:
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "boxes": [
    {
      "box": [100, 100, 200, 200]
    }
  ],
  "texts": [
    {
      "text": "person"
    }
  ]
}
```

- **响应格式**:
```json
{
  "masks": [
    {
      "label": "string",
      "mask_base64": "data:image/png;base64,..."
    }
  ],
  "message": "Success"
}
```

### 3.3 图像标注 - 自动模式
- **端点**: `POST /predict_auto`
- **功能**: 自动分割图像中的所有对象，无需任何提示
- **请求体参数**:
```json
{
  "image_base64": "字符串"      // (必需) Base64编码的图像数据
}
```

- **请求示例**:
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

- **响应格式**:
```json
{
  "masks": [
    {
      "label": "object_0",
      "mask_base64": "data:image/png;base64,..."
    }
  ],
  "message": "Successfully detected and segmented 3 objects"
}
```

### 3.4 视频标注 - 手动模式
- **端点**: `POST /predict_video`
- **功能**: 对视频进行对象跟踪，需要初始帧的框选提示
- **请求体参数**:
```json
{
  "video_base64": "字符串",    // (必需) Base64编码的视频数据
  "boxes": [                   // (必需) 初始帧的框选提示数组
    {
      "box": [x1, y1, x2, y2] // 初始边界框坐标
    }
  ]
}
```

- **请求示例**:
```json
{
  "video_base64": "JVBERi0xLjQKJcOkw7zDtsO...",
  "boxes": [
    {
      "box": [50, 50, 150, 150]
    }
  ]
}
```

- **响应格式**:
```json
{
  "frames": {
    "0": "data:image/jpeg;base64,...",
    "1": "data:image/jpeg;base64,...",
    "2": "data:image/jpeg;base64,..."
    // ... 更多帧
  },
  "debug_images": null
}
```

### 3.5 视频标注 - 自动模式
- **端点**: `POST /predict_video_auto`
- **功能**: 自动分割视频中每帧的所有对象，无需任何提示
- **请求体参数**:
```json
{
  "video_base64": "字符串"      // (必需) Base64编码的视频数据
}
```

- **请求示例**:
```json
{
  "video_base64": "JVBERi0xLjQKJcOkw7zDtsO..."
}
```

- **响应格式**:
```json
{
  "frames": {
    "0": "data:image/jpeg;base64,...",
    "1": "data:image/jpeg;base64,...",
    "2": "data:image/jpeg;base64,..."
    // ... 更多帧
  }
}
```

## 4. 数据类型定义

### 4.1 BoxPrompt (框选提示)
```json
{
  "box": [x1, y1, x2, y2]  // [左上角x, 左上角y, 右下角x, 右下角y]
}
```

### 4.2 TextPrompt (文本提示)
```json
{
  "text": "对象描述文本"
}
```

### 4.3 MaskResult (遮罩结果)
```json
{
  "label": "标签名称",
  "mask_base64": "data:image/png;base64,..."
}
```

## 5. 错误处理

### 5.1 常见错误状态码
- `400 Bad Request`: 请求参数错误
- `404 Not Found`: 请求的资源不存在
- `500 Internal Server Error`: 服务器内部错误

### 5.2 错误响应格式
```json
{
  "detail": "错误详细信息"
}
```

## 6. 使用示例

### 6.1 Python 客户端示例
```python
import requests
import base64

# 读取图像并转换为base64
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 发送请求到自动标注接口
url = "http://your-server-ip:8069/predict_auto"
payload = {
    "image_base64": image_data
}

response = requests.post(url, json=payload)
result = response.json()

print(f"检测到 {len(result['masks'])} 个对象")
for mask in result['masks']:
    print(f"标签: {mask['label']}")
```

### 6.2 JavaScript 客户端示例
```javascript
// 读取文件并转换为base64
function getBase64FromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            resolve(reader.result.split(',')[1]); // 提取base64部分
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// 调用API
async function annotateImage(file) {
    const imageBase64 = await getBase64FromFile(file);

    const response = await fetch('http://your-server-ip:8069/predict_auto', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image_base64: imageBase64
        })
    });

    const result = await response.json();
    console.log('标注结果:', result);
}
```

## 7. 部署说明

### 7.1 环境要求
- Python 3.8+
- PyTorch (支持CUDA推荐)
- Transformers库
- OpenCV (cv2)
- FastAPI
- uvicorn

### 7.2 安装依赖
```bash
pip install fastapi uvicorn opencv-python torch transformers pillow numpy
```

### 7.3 模型配置
- 将SAM3模型放置在项目根目录下的 `sam3/` 文件夹中
- 确保模型文件完整，包含必要的配置文件和权重文件

### 7.4 服务部署
推荐使用以下命令部署生产环境：
```bash
uvicorn app:app --host 0.0.0.0 --port 8069 --workers 4
```

或使用进程管理器如Supervisor或Systemd进行守护进程管理。

## 8. 性能优化说明

- 视频处理采用流式处理，避免长时间视频导致内存溢出
- 自动清理GPU缓存以优化内存使用
- 支持多线程处理提高并发性能

## 9. 注意事项

- 视频处理时间与视频长度成正比，请耐心等待处理完成
- 对于大型图像或视频，确保有足够的GPU内存
- 自动标注模式可能需要互联网连接首次下载模型
- 服务默认允许跨域请求(CORS)，生产环境中建议限制来源
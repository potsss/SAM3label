# SAM3 自动化标注服务 (SAM3 Annotation Service)

本项目是一个基于 **FastAPI** 和 **Ultralytics** 架构的自动化图像标注后端系统。它利用 **SAM3 (Segment Anything Model 3)** 的强大能力，通过标准的 HTTP 接口，为前端标注平台提供基于“点、框、文本”的多模态分割服务。

## 1. 项目概览

### 核心功能
*   **多模态提示 (Multi-modal Prompts)**：
    *   **点击 (Points)**：支持正点（目标物体）和负点（排除区域）。
    *   **框选 (Boxes)**：支持矩形框定位目标。
    *   **文本 (Texts)**：支持通过自然语言描述（如 "red car"）进行语义分割。
*   **多边形输出 (Polygon Output)**：
    *   系统不直接返回模型生成的二进制掩码（Mask），而是自动转换为**可编辑的多边形顶点列表**。
    *   内置 **Douglas-Peucker 算法**，可根据参数调整顶点的稀疏程度，极大方便了人工标注员进行二次修正。
*   **轻量化架构**：
    *   代码逻辑与模型推理分离，支持 Mock 模式（无模型文件时也能测试接口），方便本地开发。

## 2. 目录结构与模块详解

### 2.1 项目结构树
```text
SAM3code/
├── app.py                  # [入口] FastAPI 服务，处理路由和请求
├── requirements.txt        # 依赖列表
├── core/
│   ├── model_wrapper.py    # [模型层] SAM3 模型封装 (Ultralytics)，含单例与 Mock 机制
│   └── converter.py        # [核心算法] Mask 转 Polygon 转换器
├── schemas/
│   └── api.py              # [数据层] Pydantic 模型，定义接口输入输出规范
├── utils/
│   └── image_utils.py      # [工具层] 图像 Base64 解码与预处理
└── sam3_b.pt               # [权重] 模型文件 (部署时需放置在此处)
```

### 2.2 核心文件详解

*   **`app.py` (API 服务心脏)**：
    *   **职责**：负责 Web 服务的启动、路由分发、请求解析与响应封装。
    *   **关键机制**：采用单例模式加载 `SAM3Annotator`，确保模型仅在启动时初始化一次。它支持从 Base64 字符串或本地路径加载图像，并统一调度模型推理与后处理逻辑。

*   **`schemas/api.py` (数据契约)**：
    *   **职责**：使用 Pydantic 定义严谨的 API 输入输出格式。
    *   **意义**：通过 `AnnotationRequest` 提供了高度灵活的接口，允许用户混合使用点、框和文本提示。同时，它实现了自动化的输入验证，确保非法数据在进入模型逻辑前被拦截。

*   **`core/model_wrapper.py` (模型适配器)**：
    *   **职责**：封装 Ultralytics SAM3 推理逻辑，适配不同硬件环境。
    *   **核心特性**：
        *   **Mock 模式**：检测不到模型权重时自动切换到模拟推理，方便在无 GPU 环境（如笔记本）开发业务逻辑。
        *   **结果解析**：将复杂的模型输出（Tensors）转化为简单的 NumPy 掩码，并传递给转换引擎。

*   **`core/converter.py` (几何转换引擎)**：
    *   **职责**：将像素级的二进制掩码（Mask）转换为矢量级的多边形（Polygon）。
    *   **核心算法**：利用 `cv2.findContours` 提取轮廓，并通过 **Douglas-Peucker 算法** (`cv2.approxPolyDP`) 进行顶点简化。
    *   **标注友好**：该模块确保输出的多边形顶点数量适中，既保留了形状精度，又让标注员能够轻松进行人工微调。

*   **`utils/image_utils.py` (底层工具)**：
    *   **职责**：处理图像编解码。它能自动解析各种格式的 Base64 字符串（包括带 Data URI 前缀的字符串），并将其转换为 OpenCV 标准的 BGR 图像矩阵。

*   **`requirements.txt` (环境依赖)**：
    *   定义了运行本项目所需的最小环境，包括 `fastapi` (接口), `ultralytics` (模型驱动), `opencv-python` (算法处理) 等。

## 3. 安装与运行

### 3.1 环境准备
建议使用 Python 3.9+。

```bash
# 1. 克隆项目或下载代码
# 2. 安装依赖
pip install -r requirements.txt

# [服务器端] 确保安装了 CUDA 版本的 PyTorch 以启用 GPU 加速
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3.2 准备模型
请从官方渠道或 Ultralytics 仓库下载 SAM3 权重文件（如 `sam3_b.pt`），并将其放置在项目根目录下。

> **提示**：如果在本地没有模型文件，系统会自动进入 Mock 模式，返回虚拟的矩形数据，方便调试 API 连通性。

### 3.3 启动服务
```bash
python app.py
```
服务启动后默认监听：`http://0.0.0.0:8000`

## 4. 接口说明 (API Usage)

### 4.1 核心预测接口 `POST /predict`

**请求头 (Headers):** `Content-Type: application/json`

**请求体 (Body) 参数详解:**

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `image_base64` | string | 否 | 图片数据的 Base64 编码 (与 `image_path` 二选一) |
| `image_path` | string | 否 | 服务器上的绝对图片路径 (用于本地快速测试) |
| `points` | list | 否 | 点击列表，格式: `[{"point": [x, y], "label": 1}]` (1=前景, 0=背景) |
| `boxes` | list | 否 | 框选列表，格式: `[{"box": [x1, y1, x2, y2]}]` |
| `texts` | list | 否 | 文本列表，格式: `[{"text": "description"}]` |
| `epsilon_ratio` | float | 否 | 多边形简化精度，默认 `0.005`。值越小，轮廓越精细，点越多。 |

**示例 Request:**
```json
{
  "image_base64": "/9j/4AAQSkZJRg...",
  "texts": [
    {"text": "face"}
  ],
  "points": [
    {"point": [250, 300], "label": 1}
  ],
  "epsilon_ratio": 0.003
}
```

**示例 Response:**
```json
{
  "polygons": [
    {
      "points": [[240, 290], [255, 292], [260, 310], ...],
      "label": "object_0"
    }
  ],
  "message": "Success"
}
```

### 4.2 健康检查 `GET /health`
用于负载均衡器或监控脚本检查服务存活状态。

## 5. 开发与迁移指南

### 本地开发 (笔记本)
1. 编写代码逻辑（已完成）。
2. 在没有 `sam3_b.pt` 的情况下运行 `app.py`。
3. 调用接口，观察 Mock 数据返回，验证前后端数据通路。

### 服务器部署
1. 将所有代码上传至 GPU 服务器。
2. 下载真实的 SAM3 模型权重放入根目录。
3. 设置环境变量（可选）：`export SAM3_MODEL_PATH=/path/to/your/model.pt`。
4. 运行服务，Ultralytics 会自动检测并使用 GPU 进行推理。

## 6. 核心逻辑说明 (Converter)
位于 `core/converter.py` 中的 `mask_to_polygons` 函数是实现“可编辑标注”的关键：
1. **Find Contours**: 使用 OpenCV 提取二值掩码的轮廓。
2. **Approximate**: 使用 `cv2.approxPolyDP` 对轮廓进行多边形拟合。这不仅减少了传输的数据量，更重要的是生成的“关键点”非常适合人工在标注工具中进行拖拽修改。

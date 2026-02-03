# SAM3 图像标注服务 API 文档

## 1. 简介

本 API 文档描述了如何使用 SAM3 图像标注服务。该服务基于 FastAPI 构建，提供了对图像和视频进行目标分割和追踪的功能。您可以根据**边界框 (Box)** 或**文本提示 (Text Prompt)** 来进行标注。

**基础 URL**: `http://<your-server-ip>:8069`

## 2. 通用信息

- **CORS**: 服务已启用 CORS 中间件，允许来自任何源 (`*`) 的跨域请求。
- **数据格式**: 所有请求和响应主体均为 JSON 格式。
- **图片/视频编码**: 服务接受 Base64 编码的字符串作为图片或视频输入。

---

## 3. 健康检查 (`/health`)

用于检查服务是否正在运行以及模型是否已成功加载。

- **路径**: `/health`
- **方法**: `GET`
- **请求体**: 无
- **成功响应 (200 OK)**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```
- **说明**: 如果 `model_loaded` 为 `false`，表示模型文件加载失败，服务无法处理标注请求。

---

## 4. 图像标注 (`/predict`)

对单张静态图片进行目标分割。支持使用边界框或文本作为提示。

- **路径**: `/predict`
- **方法**: `POST`
- **请求体 (`AnnotationRequest`)**:

  | 字段 | 类型 | 是否必须 | 描述 |
  | :--- | :--- | :--- | :--- |
  | `image_base64` | string | 否 | Base64 编码的图像文件字符串。 |
  | `image_path` | string | 否 | 服务器上的图像文件绝对路径。`image_base64` 和 `image_path` 必须提供一个。 |
  | `boxes` | array | 否 | 边界框提示列表。参见 `BoxPrompt` 模型。 |
  | `texts` | array | 否 | 文本提示列表。参见 `TextPrompt` 模型。 |

  **`BoxPrompt` 模型**:
  ```json
  {
    "box": "[x1, y1, x2, y2]"
  }
  ```
  **`TextPrompt` 模型**:
  ```json
  {
    "text": "描述物体的文本"
  }
  ```

- **成功响应 (`AnnotationResponse`)**:
  ```json
  {
    "masks": [
      {
        "label": "person_1",
        "mask_base64": "<base64-encoded-mask-png>"
      }
    ],
    "message": "Success"
  }
  ```
- **异常响应**:
  - `400 Bad Request`: 如果 `image_base64` 和 `image_path` 均未提供，或 Base64 格式无效。
  - `404 Not Found`: 如果 `image_path` 指定的文件不存在。
  - `500 Internal Server Error`: 如果在模型推理过程中发生错误。

- **请求示例**:
  ```json
  {
    "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
    "boxes": [
      { "box": [150.5, 200.0, 450.2, 600.8] }
    ],
    "texts": [
      { "text": "a red car" }
    ]
  }
  ```

---

## 5. 视频标注 (框选追踪) (`/predict_video`)

在视频中追踪由初始边界框指定的目标。

- **路径**: `/predict_video`
- **方法**: `POST`
- **请求体 (`VideoAnnotationRequest`)**:

  | 字段 | 类型 | 是否必须 | 描述 |
  | :--- | :--- | :--- | :--- |
  | `video_base64` | string | 是 | Base64 编码的视频文件。 |
  | `boxes` | array | 是 | 在视频第一帧上指定的初始边界框列表。 |

- **成功响应 (`VideoAnnotationResponse`)**:
  ```json
  {
    "frames": {
      "0": "<base64-annotated-frame0-png>",
      "1": "<base64-annotated-frame1-png>",
      ...
    },
    "debug_images": null
  }
  ```
  - `frames`: 一个字典，键为帧的序号（字符串格式），值为标注后的该帧画面的 Base64 编码。

- **异常响应**:
  - `400 Bad Request`: 如果未提供 `boxes`。
  - `500 Internal Server Error`: 如果视频处理或模型推理失败。

- **请求示例**:
  ```json
  {
    "video_base64": "AAAAGGZ0eX...',
    "boxes": [
      { "box": [100, 150, 250, 300] }
    ]
  }
  ```

---

## 6. 视频标注 (文本追踪) (`/predict_video_text`)

在视频中检测和追踪与给定文本描述匹配的所有目标。

- **路径**: `/predict_video_text`
- **方法**: `POST`
- **请求体 (`VideoTextTrackingRequest`)**:

  | 字段 | 类型 | 是否必须 | 描述 |
  | :--- | :--- | :--- | :--- |
  | `video_base64` | string | 是 | Base64 编码的视频文件。 |
  | `text_prompt` | string | 是 | 要在视频中检测和追踪的物体的文本描述，例如 "一个人", "车", "狗"。 |

- **成功响应 (`VideoTextTrackingResponse`)**:
  ```json
  {
    "frames": {
      "0": "<base64-annotated-frame0-png>",
      "1": "<base64-annotated-frame1-png>",
      ...
    }
  }
  ```
  - `frames`: 字典结构，键为帧序号，值为包含分割掩码的标注后帧画面的 Base64 编码。

- **异常响应**:
  - `400 Bad Request`: 如果 `text_prompt` 为空。
  - `500 Internal Server Error`: 如果视频处理或模型推理失败。

- **请求示例**:
  ```json
  {
    "video_base64": "AAAAGGZ0eX...',
    "text_prompt": "a running dog"
  }
  ```

---

## 7. 数据模型 (Schemas)

### `BoxPrompt`
描述一个矩形边界框。

- `box` (List[float]): 包含四个浮点数的列表 `[x1, y1, x2, y2]`，分别代表左上角和右下角的坐标。

### `TextPrompt`
描述一个文本提示。

- `text` (string): 描述目标的文本。

### `MaskResult`
描述单个分割结果。

- `label` (string): 唯一标识符，例如 `person_1`。
- `mask_base64` (string): 分割结果掩码图的 Base64 编码 (PNG 格式)。

### `AnnotationResponse`
`/predict` 端点的响应模型。

- `masks` (List[`MaskResult`]): 分割结果列表。
- `message` (string): 状态消息，默认为 "Success"。

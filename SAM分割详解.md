# SAM-Med2D 医学图像分割详解

## 目录
1. [SAM概述](#1-sam概述)
2. [SAM-Med2D架构](#2-sam-med2d架构)
3. [核心组件详解](#3-核心组件详解)
4. [训练策略与微调技术](#4-训练策略与微调技术)
5. [项目实现解析](#5-项目实现解析)
6. [使用示例](#6-使用示例)

---

## 1. SAM概述

### 1.1 什么是SAM？

SAM（Segment Anything Model）是由Meta AI提出的一种通用图像分割模型，被称为"分割一切模型"。它最大的特点是** prompt-based learning（基于提示的学习）**，能够根据各种提示（点、框、文本等）完成分割任务。

### 1.2 SAM-Med2D的独特之处

SAM-Med2D是针对**医学2D图像**优化的SAM变体，具有以下特点：

| 特性 | 原始SAM | SAM-Med2D |
|------|---------|-----------|
| 训练数据 | 自然图像 | 医学图像（CT、MRI、X光等） |
| 预训练策略 | 大量自然图像 | 医学图像领域自适应 |
| 提示方式 | 点、框、文本 | 医学特有的标注方式 |
| 性能优化 | 通用分割 | 医学图像精准分割 |

---

## 2. SAM-Med2D架构

### 2.1 整体架构图

```
输入图像 → [图像编码器] → 图像嵌入
                ↓
用户提示（点/框）→ [提示编码器] → 提示嵌入
                ↓
        [掩码解码器] ← 图像嵌入 + 提示嵌入
                ↓
            输出掩码
```

### 2.2 三大核心组件

SAM-Med2D由三个核心组件构成：

1. **图像编码器（Image Encoder）**：使用ViT（Vision Transformer）提取图像特征
2. **提示编码器（Prompt Encoder）**：编码用户提供的点、框等提示信息
3. **掩码解码器（Mask Decoder）**：结合图像和提示信息生成最终分割掩码

---

## 3. 核心组件详解

### 3.1 图像编码器（Image Encoder）

#### 3.1.1 结构

图像编码器使用**Vision Transformer (ViT)**架构：

```python
# 项目中的实现（predictor_sammed.py）
self.features = self.model.image_encoder(input_image.to(self.device))
```

#### 3.1.2 图像预处理流程

```python
# 1. 像素值归一化
input_image = (image - pixel_mean) / pixel_std

# 2. 调整图像大小到固定尺寸（如256x256）
resized_image = cv2.resize(
    input_image,
    (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
    interpolation=cv2.INTER_NEAREST
)

# 3. 转换为CHW格式
resized_image = np.transpose(resized_image, (0, 3, 1, 2))
```

#### 3.1.3 独特设计

- **Image Encoder Adapter**：适配器模块，用于医学图像特征增强
- **双线性插值**：保持边缘信息，减少分割精度损失
- **固定尺寸输入**：将不同尺寸的医学图像统一调整为固定大小

### 3.2 提示编码器（Prompt Encoder）

#### 3.2.1 支持的提示类型

| 提示类型 | 说明 | 在本项目中的应用 |
|----------|------|------------------|
| 点（Point） | 正点（前景）或负点（背景） | 点标注分割 |
| 框（Box） | 边界框，定义分割区域 | 框标注分割 |

#### 3.2.2 点的编码

```python
# 点的坐标和标签处理
point_coords = self.apply_coords(point_coords, self.original_size, self.new_size)
coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
```

#### 3.2.3 框的编码

```python
# 框的坐标处理
box = self.apply_boxes(box, self.original_size, self.new_size)
box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
```

### 3.3 掩码解码器（Mask Decoder）

#### 3.3.1 解码器结构

掩码解码器是SAM的核心，负责生成最终的分割掩码：

```
图像嵌入 + 提示嵌入 → Transformer块 → 掩码预测
                           ↓
                      IoU预测头
```

#### 3.3.2 多掩码输出（Multimask Output）

SAM的一个独特设计是**输出多个候选掩码**：

```python
masks, iou_predictions, low_res_masks = self.predict_torch(
    coords_torch,
    labels_torch,
    box_torch,
    mask_input_torch,
    multimask_output=True,  # 开启多掩码输出
)
```

#### 3.3.3 掩码后处理

```python
def postprocess_masks(self, low_res_masks, image_size, original_size):
    # 1. 从低分辨率上采样到模型输入尺寸
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False
    )
    # 2. 再上采样到原始图像尺寸
    masks = F.interpolate(
        masks,
        original_size,
        mode="bilinear",
        align_corners=False
    )
    return masks
```

#### 3.3.4 二值化处理

```python
# 使用sigmoid激活后进行二值化
sigmoid_output = torch.sigmoid(masks)
masks = (sigmoid_output > 0.5).float()  # 阈值0.5
```

---

## 4. 训练策略与微调技术

### 4.1 预训练策略

#### 4.1.1 基础预训练

SAM-Med2D的预训练分为两个阶段：

**第一阶段：大规模自然图像预训练**
- 使用SA-1B数据集（1100万张图像）
- 学习通用的分割能力
- 获得强大的图像特征表示

**第二阶段：医学图像微调**
- 使用医学图像数据集（如3D Medical Segmentation数据集）
- 领域自适应训练
- 医学图像特有特征学习

#### 4.1.2 提示预训练

在预训练阶段，SAM使用**随机提示策略**：

1. **几何提示**：随机生成点、框
2. **语义提示**：模拟用户交互
3. **模糊提示**：处理不完整的标注

### 4.2 微调技术（Fine-tuning）

#### 4.2.1 编码器微调

本项目中可以选择性地微调图像编码器：

```python
# 在predictor中使用编码器
self.features = self.model.image_encoder(input_image.to(self.device))
```

#### 4.2.2 Adapter机制

SAM-Med2D的一个独特设计是**Adapter（适配器）**：

```python
# 项目中使用的配置
args = Namespace(
    image_size=256,
    encoder_adapter=True,  # 启用适配器机制
    sam_checkpoint="./sam-med2d_refine.pth"
)
```

**Adapter的作用**：
- 在不破坏预训练权重的情况下学习医学图像特征
- 参数高效微调（Parameter Efficient Fine-tuning）
- 保留原始SAM的泛化能力

#### 4.2.3 轻量化微调策略

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| Full Fine-tuning | 全部参数微调 | 数据充足时 |
| Adapter Tuning | 只训练适配器 | 医学图像适配 |
| Linear Probing | 只训练分类头 | 快速迁移 |

### 4.3 损失函数设计

#### 4.3.1 分割损失

SAM使用的损失函数组合：

```
总损失 = Dice损失 + Focal损失
```

- **Dice Loss**：处理类别不平衡
- **Focal Loss**：关注难分割区域

#### 4.3.2 IoU损失

额外使用IoU（Intersection over Union）损失提升分割精度：

```python
# IoU预测损失
iou_predictions = self.model.mask_decoder(...)
```

---

## 5. 项目实现解析

### 5.1 核心文件结构

```
PythonProject8/
├── sam_med2d_funcs.py      # 分割函数（实际使用）
├── sam_med2d.py            # 分割函数（备用）
├── segment_anything/
│   ├── predictor_sammed.py # SAMmed预测器类
│   └── build_sam.py        # 模型构建
└── sam-med2d_refine.pth    # 训练好的权重
```

### 5.2 分割函数解析

#### 5.2.1 点分割（Point Segmentation）

```python
def point_segmentation(model, input_points, input_labels, image_array, file_type):
    # 1. 创建预测器
    predictor = SammedPredictor(model)

    # 2. 图像预处理（灰度转RGB）
    image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    # 3. 设置图像（提取特征）
    predictor.set_image(image)

    # 4. 执行预测
    masks, scores, logits = predictor.predict(
        point_coords=np.array(input_points),      # 点坐标
        point_labels=np.array(input_labels),      # 点标签（1=前景，0=背景）
        multimask_output=True,                     # 输出多个掩码
    )

    # 5. 合并多个掩码
    temp = 0
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            temp += mask.cpu().numpy()
        else:
            temp += mask

    return temp
```

**关键点**：
- `input_labels=1` 表示正样本点（目标区域）
- `input_labels=0` 表示负样本点（非目标区域）
- `multimask_output=True` 返回多个候选掩码

#### 5.2.2 单框分割（Single Box Segmentation）

```python
def single_box_segmentation(model, box_points, image_array):
    predictor = SammedPredictor(model)
    image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    predictor.set_image(image)

    # 定义边界框 [x1, y1, x2, y2]
    input_box = np.array([box_points[0], box_points[1],
                          box_points[2], box_points[3]])

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True,
    )

    return masks[0]  # 返回第一个（最佳）掩码
```

#### 5.2.3 多框分割（Multiple Box Segmentation）

```python
def multiple_box_segmentation(model, box_points, image_array, args):
    predictor = SammedPredictor(model)
    image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    predictor.set_image(image)

    # 批量处理多个框
    input_boxes = torch.tensor(np.array(box_points), device=predictor.device)

    # 应用框变换（适配模型输入尺寸）
    transformed_boxes = predictor.apply_boxes_torch(
        input_boxes,
        image.shape[:2],
        (args.image_size, args.image_size)
    )

    # GPU加速的批量预测
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )

    # 合并所有框的掩码
    temp = 0
    for mask in masks:
        temp = temp + mask.cpu().numpy()

    return temp[0]
```

### 5.3 图像预处理详解

#### 5.3.1 尺寸适配

```python
# 原始图像尺寸
ori_h, ori_w, _ = input_image.shape

# 目标尺寸（模型输入）
new_size = (self.model.image_encoder.img_size,
            self.model.image_encoder.img_size)

# 坐标缩放比例
coords[..., 0] = coords[..., 0] * (new_w / old_w)
coords[..., 1] = coords[..., 1] * (new_h / old_h)
```

#### 5.3.2 像素值归一化

```python
# 使用模型的均值和标准差
pixel_mean = self.model.pixel_mean.squeeze().cpu().numpy()
pixel_std = self.model.pixel_std.squeeze().cpu().numpy()

# 归一化
input_image = (image - pixel_mean) / pixel_std
```

### 5.4 批量推理优化

#### 5.4.1 多框批量处理

SAM-Med2D支持高效的批量推理：

```python
# 逐个处理每个框
for i in range(boxes.shape[0]):
    pre_boxes = boxes[i:i+1,...]

    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
        points=points,
        boxes=pre_boxes,
        masks=mask_input,
    )

    low_res_masks, iou_predictions = self.model.mask_decoder(...)

    # 收集结果
    mask_list.append(pre_masks)

# 合并结果
masks = torch.cat(mask_list, dim=0)
```

#### 5.4.2 IoU选择

当输出多个掩码时，选择IoU最高的：

```python
if multimask_output:
    max_values, max_indexs = torch.max(iou_predictions, dim=1)
    max_values = max_values.unsqueeze(1)
    iou_predictions = max_values
    low_res_masks = low_res_masks[:, max_indexs]
```

---

## 6. 使用示例

### 6.1 基本使用流程

```python
# 1. 加载模型
from segment_anything import sam_model_registry

sam = sam_model_registry["vit_b"](
    checkpoint="./sam-med2d_refine.pth"
)
sam.to(device="cuda")
sam.eval()

# 2. 准备图像
image_array = ...  # 医学图像数组

# 3. 点分割
input_points = [[x, y]]       # 坐标
input_labels = [1]             # 1=前景点

result_mask = point_segmentation(
    model=sam,
    input_points=input_points,
    input_labels=input_labels,
    image_array=image_array,
    file_type="DICOM"
)

# 4. 框分割
box_points = [x1, y1, x2, y2]  # 左上角和右下角坐标

result_mask = single_box_segmentation(
    model=sam,
    box_points=box_points,
    image_array=image_array
)
```

### 6.2 在项目中集成

```python
# 在SAMMed_Viewer.py中的使用
def on_action_startSegmentation(self):
    # 获取点标注数据
    points_dict = getPointsDict()

    for index, data in points_dict.items():
        # 获取切片
        index_z = self.dicomdata.shape[2] - int(index) - 1
        select_layer_image = self.dicomdata[:, :, index_z]

        # 获取点和标签
        input_points = []
        input_labels = data["label"]
        for point in data["points"]:
            input_points.append([point[0], point[1]])

        # 执行分割
        temp = point_segmentation(
            self.model,
            input_points,
            input_labels,
            select_layer_image,
            self.dataformat
        )

        # 保存结果
        self.segmentation_Result[:, :, index_z] = temp.T
```

---

## 7. 总结

### 7.1 SAM-Med2D的独特优势

1. **提示驱动的分割**：通过点、框等提示实现精准分割
2. **多掩码输出**：提供多个候选掩码，提高鲁棒性
3. **Adapter机制**：高效的医学图像领域自适应
4. **统一的分割框架**：一个模型解决多种分割任务

### 7.2 训练微调关键点

1. **两阶段训练**：先在大规模数据预训练，再在医学数据微调
2. **参数高效微调**：Adapter技术保护预训练知识
3. **多任务联合**：点分割+框分割联合训练
4. **IoU引导优化**：通过IoU预测头提升分割质量

### 7.3 应用场景

- 医学影像诊断辅助
- CT/MRI图像器官分割
- 肿瘤区域检测
- 手术规划图像分析

---

*文档生成日期：2026-03-24*
*项目：MedSAMate - SAM-Med2D医学图像分割平台*

# extensions/Cutout-processing/scripts/Cutout.py
import os
import cv2
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Tuple
import gradio as gr
import platform
from modules import scripts, shared

# 全局配置
MODELS_DIR = os.path.join(scripts.basedir(), "models")
OUTPUT_DIR = os.path.join(scripts.basedir(), "AI_Cutout_Results")

# 模型配置
U2NET_MODEL = "u2net.onnx"
MODNET_MODEL = "modnet_photographic_portrait_matting.onnx"
ISNET_ANIME_MODEL = "isnet-anime.onnx"

class AIImageMatting(scripts.Script):
    """AI图像抠图主类"""
    def __init__(self):
        super().__init__()
        self.universal = UniversalCutout()
        self.portrait = PortraitCutout()
        self.anime = AnimeCutout()

    def title(self):
        return "🎭 AI Image Matting"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("🎭 AI Image Matting", open=False):
            with gr.Tabs():
                # 通用物体抠图标签页
                with gr.TabItem("🔍 通用物体抠图"):
                    gr.Markdown("### 通用物体抠图（支持任意物体）")
                    with gr.Row():
                        self.universal_input = gr.Image(label="输入图像", type="pil", height=300)
                        self.universal_output = gr.Image(label="透明背景结果", type="pil", interactive=False, height=300)
                    with gr.Row():
                        universal_process = gr.Button("✨ 开始抠图", variant="primary")
                        universal_mask_btn = gr.Button("🎭 生成蒙版")
                        universal_open_btn = gr.Button("📂 打开目录")
                    self.universal_status = gr.Textbox(label="处理状态", interactive=False)

                # 专业人像抠图标签页
                with gr.TabItem("👤 专业人像抠图"):
                    gr.Markdown("### 专业人像抠图（优化人像细节）")
                    with gr.Row():
                        self.portrait_input = gr.Image(label="上传人像", type="pil", height=300)
                        self.portrait_output = gr.Image(label="透明背景结果", type="pil", interactive=False, height=300)
                    with gr.Row():
                        portrait_process = gr.Button("✨ 开始处理", variant="primary")
                        portrait_mask_btn = gr.Button("🎭 生成蒙版")
                        portrait_open_btn = gr.Button("📂 打开目录")
                    with gr.Accordion("⚙️ 高级参数设置", open=False):
                        with gr.Row():
                            self.blur_radius = gr.Slider(0, 15, 5, step=1, label="边缘柔化半径")
                            self.edge_strength = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="边缘增强强度")
                        with gr.Row():
                            self.erode_strength = gr.Slider(-10, 10, 0, step=1, label="腐蚀/膨胀调整")
                            portrait_reset_btn = gr.Button("🔄 恢复默认", variant="secondary")
                    self.portrait_status = gr.Textbox(label="处理状态", interactive=False)

                # 卡通动漫抠图标签页
                with gr.TabItem("🧙 卡通动漫抠图"):
                    gr.Markdown("### 卡通动漫抠图（优化动漫角色）")
                    with gr.Row():
                        self.anime_input = gr.Image(label="上传动漫图像", type="pil", height=300)
                        self.anime_output = gr.Image(label="透明背景结果", type="pil", interactive=False, height=300)
                    with gr.Row():
                        anime_process = gr.Button("✨ 开始抠图", variant="primary")
                        anime_mask_btn = gr.Button("🎭 生成蒙版")
                        anime_open_btn = gr.Button("📂 打开目录")
                    with gr.Accordion("⚙️ 高级参数设置", open=False):
                        with gr.Row():
                            self.anime_blur = gr.Slider(0, 10, 1, step=1, label="边缘柔化强度")
                            self.anime_threshold = gr.Slider(0, 255, 148, step=1, label="蒙版阈值")
                        with gr.Row():
                            self.anime_erode = gr.Slider(-5, 5, -1, step=1, label="腐蚀/膨胀调整")
                            anime_reset_btn = gr.Button("🔄 恢复默认", variant="secondary")
                    self.anime_status = gr.Textbox(label="处理状态", interactive=False)

        # 通用抠图事件绑定
        universal_process.click(
            self.universal.process_image,
            inputs=[self.universal_input],
            outputs=[self.universal_output, self.universal_status]
        )
        universal_mask_btn.click(
            self.generate_mask,
            inputs=[self.universal_input, gr.State(False), gr.State(None)],
            outputs=[self.universal_output, self.universal_status]
        )
        universal_open_btn.click(lambda: open_folder(OUTPUT_DIR), outputs=[self.universal_status])

        # 人像抠图事件绑定
        portrait_process.click(
            self.portrait.process_image,
            inputs=[self.portrait_input, self.blur_radius, self.edge_strength, self.erode_strength],
            outputs=[self.portrait_output, self.portrait_status]
        )
        portrait_mask_btn.click(
            self.generate_mask,
            inputs=[self.portrait_input, gr.State(True), gr.State(None)],
            outputs=[self.portrait_output, self.portrait_status]
        )
        portrait_open_btn.click(lambda: open_folder(OUTPUT_DIR), outputs=[self.portrait_status])
        portrait_reset_btn.click(
            lambda: [5, 1.0, 0],
            outputs=[self.blur_radius, self.edge_strength, self.erode_strength]
        )

        # 动漫抠图事件绑定
        anime_process.click(
            self.anime.process_image,
            inputs=[self.anime_input, self.anime_blur, self.anime_threshold, self.anime_erode],
            outputs=[self.anime_output, self.anime_status]
        )
        anime_mask_btn.click(
            self.generate_mask,
            inputs=[self.anime_input, gr.State(False), gr.State(True)],
            outputs=[self.anime_output, self.anime_status]
        )
        anime_open_btn.click(lambda: open_folder(OUTPUT_DIR), outputs=[self.anime_status])
        anime_reset_btn.click(
            lambda: [1, 148, -1],
            outputs=[self.anime_blur, self.anime_threshold, self.anime_erode]
        )

        return [self.universal_input, self.portrait_input, self.anime_input]

    def generate_mask(self, img, is_portrait, is_anime):
        """生成黑白蒙版"""
        try:
            start_time = time.time()
            if is_portrait:
                self.portrait.load_model()
                mask = self.portrait.process_mask(img)
                processor = "MODNet"
            elif is_anime:
                self.anime.load_model()
                mask = self.anime.process_mask(img)
                processor = "ISNet Anime"
            else:
                self.universal.initialize_model()
                mask = self.universal.process_mask(img)
                processor = "U2Net"

            save_path = save_result(mask)
            return mask, f"✅ {processor}蒙版生成 | 耗时: {time.time()-start_time:.1f}s"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"

class UniversalCutout:
    """通用物体抠图处理器"""
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def initialize_model(self):
        if self.session is None:
            model_path = os.path.join(MODELS_DIR, U2NET_MODEL)
            self.validate_model(model_path)
            
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            except Exception as e:
                raise RuntimeError(f"模型初始化失败: {str(e)}")

    def validate_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件缺失: {os.path.basename(path)}")

    def preprocess(self, img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
        img = img.convert("RGB")
        original_size = img.size
        img_arr = np.array(img)
        
        h, w = img_arr.shape[:2]
        target_size = 320
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img_arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized.astype(np.float32) / 255.0
        
        padded -= np.array(self.norm_mean)
        padded /= np.array(self.norm_std)
        
        return np.transpose(padded, (2, 0, 1))[np.newaxis, ...], original_size, (x_offset, y_offset, new_w, new_h)

    def postprocess(self, pred: np.ndarray, original_size: Tuple[int, int], padding_info: Tuple[int, int, int, int]) -> Image.Image:
        pred = 1 / (1 + np.exp(-pred[0][0]))
        mask = (pred * 255).astype(np.uint8)
        
        x_offset, y_offset, new_w, new_h = padding_info
        valid_mask = mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        
        h, w = original_size[1], original_size[0]
        resized_mask = cv2.resize(valid_mask, (w, h), interpolation=cv2.INTER_CUBIC)
        
        _, binary_mask = cv2.threshold(resized_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary_mask)

    def process_image(self, img: Image.Image):
        try:
            start_time = time.time()
            self.initialize_model()
            
            input_tensor, original_size, padding_info = self.preprocess(img)
            pred = self.session.run(
                [self.output_name], 
                {self.input_name: input_tensor}
            )[0]
            
            mask = self.postprocess(pred, original_size, padding_info)
            result = create_alpha_image(img, mask)
            save_path = save_result(result)
            
            return result, f"✅ 抠图成功 | 耗时: {time.time()-start_time:.1f}s | 路径: {save_path}"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"

    def process_mask(self, img: Image.Image):
        self.initialize_model()
        input_tensor, original_size, padding_info = self.preprocess(img)
        pred = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        return self.postprocess(pred, original_size, padding_info)

class PortraitCutout:
    """专业人像抠图处理器"""
    def __init__(self):
        self.model = None
        self.last_result = None

    def load_model(self):
        if self.model is None:
            model_path = os.path.join(MODELS_DIR, MODNET_MODEL)
            self.validate_model(model_path)
            
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.model = ort.InferenceSession(model_path, providers=providers)
            except Exception as e:
                raise RuntimeError(f"模型加载失败: {str(e)}")

    def validate_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件缺失: {os.path.basename(path)}")

    def process_image(self, img, blur, edge, erode):
        try:
            start_time = time.time()
            original = img.copy()
            mask = self.process_mask(img, blur, erode)
            result = create_alpha_image(original, mask)
            save_path = save_result(result)
            return result, f"✅ 人像抠图完成 | 耗时: {time.time()-start_time:.1f}s"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"

    def process_mask(self, img, blur=5, erode=0):
        self.load_model()
        img_arr = np.array(img.convert("RGB"))
        h, w = img_arr.shape[:2]
        
        processed = cv2.resize(img_arr, (512, 512), interpolation=cv2.INTER_AREA)
        processed = (processed / 255.0).astype(np.float32)
        input_tensor = np.transpose(processed, (2, 0, 1))[None, ...]
        
        output = self.model.run(None, {'input': input_tensor})[0][0][0]
        matte = (output * 255).astype(np.uint8)
        matte = cv2.resize(matte, (w, h))
        
        if blur > 0:
            matte = cv2.GaussianBlur(matte, (blur*2+1,)*2, 0)
        if erode != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            matte = cv2.erode(matte, kernel, iterations=abs(erode)) if erode <0 else cv2.dilate(matte, kernel, iterations=erode)
            
        return Image.fromarray(matte)

class AnimeCutout:
    """卡通动漫抠图处理器"""
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def load_model(self):
        if self.session is None:
            model_path = os.path.join(MODELS_DIR, ISNET_ANIME_MODEL)
            self.validate_model(model_path)
            
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            except Exception as e:
                raise RuntimeError(f"模型加载失败: {str(e)}")

    def validate_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件缺失: {os.path.basename(path)}")

    def preprocess(self, img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        img = img.convert("RGB")
        original_size = img.size
        img_arr = np.array(img)
        
        # ISNet Anime 模型需要1024x1024输入
        target_size = 1024
        h, w = img_arr.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img_arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized.astype(np.float32) / 255.0
        
        padded -= np.array(self.norm_mean)
        padded /= np.array(self.norm_std)
        
        return np.transpose(padded, (2, 0, 1))[np.newaxis, ...], original_size, (x_offset, y_offset, new_w, new_h)

    def postprocess(self, pred: np.ndarray, original_size: Tuple[int, int], padding_info: Tuple[int, int, int, int], 
                   threshold: int = 148, blur: int = 1, erode: int = -1) -> Image.Image:
        pred = 1 / (1 + np.exp(-pred[0][0]))
        mask = (pred * 255).astype(np.uint8)
        
        x_offset, y_offset, new_w, new_h = padding_info
        valid_mask = mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        
        h, w = original_size[1], original_size[0]
        resized_mask = cv2.resize(valid_mask, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 应用阈值
        _, binary_mask = cv2.threshold(resized_mask, threshold, 255, cv2.THRESH_BINARY)
        
        # 应用边缘柔化
        if blur > 0:
            binary_mask = cv2.GaussianBlur(binary_mask, (blur*2+1, blur*2+1), 0)
        
        # 应用腐蚀/膨胀
        if erode != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.erode(binary_mask, kernel, iterations=abs(erode)) if erode < 0 else cv2.dilate(binary_mask, kernel, iterations=erode)
        
        return Image.fromarray(binary_mask)

    def process_image(self, img: Image.Image, blur: int = 1, threshold: int = 148, erode: int = -1):
        try:
            start_time = time.time()
            self.load_model()
            
            input_tensor, original_size, padding_info = self.preprocess(img)
            pred = self.session.run(
                [self.output_name], 
                {self.input_name: input_tensor}
            )[0]
            
            mask = self.postprocess(pred, original_size, padding_info, threshold, blur, erode)
            result = create_alpha_image(img, mask)
            save_path = save_result(result)
            
            return result, f"✅ 动漫抠图完成 | 耗时: {time.time()-start_time:.1f}s | 路径: {save_path}"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"

    def process_mask(self, img: Image.Image, blur: int = 1, threshold: int = 148, erode: int = -1):
        self.load_model()
        input_tensor, original_size, padding_info = self.preprocess(img)
        pred = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        return self.postprocess(pred, original_size, padding_info, threshold, blur, erode)

def create_alpha_image(img: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    rgba.putalpha(mask)
    return rgba

def save_result(image: Image.Image) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"cutout_{int(time.time()*1000)}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    image.save(save_path)
    return save_path

def open_folder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            
        system = platform.system()
        path = os.path.normpath(path)
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
        return "📂 输出目录已打开"
    except Exception as e:
        return f"⚠️ 打开失败: {str(e)}"

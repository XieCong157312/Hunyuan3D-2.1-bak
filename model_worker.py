"""Memory-sequential ModelWorker for Hunyuan3D-2.1 (24GB-class GPUs)."""
import os
import time
import uuid
import base64
import gc
import threading
from io import BytesIO

from PIL import Image
import torch

import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found; continuing without compatibility fix")
except Exception as e:
    print(f"Warning: torchvision compatibility fix failed: {e}")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


def release_cuda_memory():
    """Release unreferenced CUDA allocations before loading the next stage."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg'),
    }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def load_image_from_base64(image):
    # Keep the decoded image detached from the BytesIO lifetime.
    with Image.open(BytesIO(base64.b64decode(image))) as im:
        return im.copy()


class ModelWorker:
    """One GPU worker. Shape and Paint are never resident together."""

    def __init__(self, model_path='tencent/Hunyuan3D-2.1',
                 subfolder='hunyuan3d-dit-v2-1', device='cuda',
                 low_vram_mode=True, worker_id=None, model_semaphore=None,
                 save_dir='gradio_cache', mc_algo='mc',
                 enable_flashvdm=False, compile=False,
                 paint_max_num_view=4, paint_resolution=512):
        self.model_path = model_path
        self.subfolder = subfolder
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        self.mc_algo = mc_algo
        self.enable_flashvdm = enable_flashvdm
        self.compile = compile
        self.paint_max_num_view = paint_max_num_view
        self.paint_resolution = paint_resolution
        self.pipeline = None
        self.paint_pipeline = None
        self._gpu_lock = threading.RLock()

        os.makedirs(self.save_dir, exist_ok=True)
        self.rembg = BackgroundRemover()
        logger.info("Worker %s initialized; GPU pipelines will be loaded on demand", self.worker_id)

        # Preserve original behavior of clearing regular files in the cache, but
        # do not fail on nested directories.
        for name in os.listdir(self.save_dir):
            path = os.path.join(self.save_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Could not remove old cache file: %s", path)

    def _load_shape(self):
        if self.pipeline is None:
            logger.info("Loading Shape pipeline")
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_path)
            if self.enable_flashvdm:
                algo = 'mc' if self.device in ('cpu', 'mps') else self.mc_algo
                self.pipeline.enable_flashvdm(mc_algo=algo)
            if self.compile:
                self.pipeline.compile()
        return self.pipeline

    def _load_paint(self):
        if self.paint_pipeline is None:
            logger.info("Loading Paint pipeline (views=%s, resolution=%s)",
                        self.paint_max_num_view, self.paint_resolution)
            conf = Hunyuan3DPaintConfig(self.paint_max_num_view, self.paint_resolution)
            conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        return self.paint_pipeline

    def _unload_shape(self):
        if self.pipeline is not None:
            logger.info("Unloading Shape pipeline")
            pipe, self.pipeline = self.pipeline, None
            del pipe
        release_cuda_memory()

    def _unload_paint(self):
        if self.paint_pipeline is not None:
            logger.info("Unloading Paint pipeline")
            pipe, self.paint_pipeline = self.paint_pipeline, None
            del pipe
        release_cuda_memory()

    def get_queue_length(self):
        if self.model_semaphore is None:
            return 0
        return (self.model_semaphore._value if hasattr(self.model_semaphore, '_value') else 0) + \
               (len(self.model_semaphore._waiters) if getattr(self.model_semaphore, '_waiters', None) is not None else 0)

    def get_status(self):
        return {"speed": 1, "queue_length": self.get_queue_length()}

    @torch.inference_mode()
    def generate(self, uid, params):
        # A single lock spans Shape, model teardown, Paint, and final conversion.
        # This prevents two requests from interleaving model residency on one GPU.
        with self._gpu_lock:
            return self._generate_sequential(uid, params)

    def _generate_sequential(self, uid, params):
        start_time = time.time()
        logger.info("Generating 3D model for uid: %s", uid)
        if not params.get('image'):
            raise ValueError("No input image provided")
        image = load_image_from_base64(params['image'])
        was_rgb = image.mode == 'RGB'
        image = image.convert('RGBA')
        if was_rgb:
            image = self.rembg(image.convert('RGB')).convert('RGBA')

        initial_save_path = os.path.join(self.save_dir, f'{uid}_initial.glb')
        try:
            pipe = self._load_shape()
            mesh = pipe(image=image)[0]
            mesh.export(initial_save_path)
            del mesh
            logger.info("Shape stage completed in %.2fs", time.time() - start_time)
        except Exception as e:
            logger.exception("Shape generation failed")
            raise ValueError(f"Failed to generate 3D mesh: {e}") from e
        finally:
            # Shape must be fully unloaded before Paint is constructed.
            self._unload_shape()

        final_save_path = initial_save_path
        try:
            paint = self._load_paint()
            output_obj = os.path.join(self.save_dir, f'{uid}_texturing.obj')
            textured_obj = paint(
                mesh_path=initial_save_path,
                image_path=image,
                output_mesh_path=output_obj,
                save_glb=False,
            )
            glb_tmp = os.path.join(self.save_dir, f'{uid}_texturing.glb')
            quick_convert_with_obj2gltf(textured_obj, glb_tmp)
            final_save_path = os.path.join(self.save_dir, f'{uid}_textured.glb')
            os.replace(glb_tmp, final_save_path)
            logger.info("Paint stage completed in %.2fs", time.time() - start_time)
        except Exception as e:
            logger.exception("Texture generation failed; returning untextured Shape mesh")
            final_save_path = initial_save_path
        finally:
            self._unload_paint()
            del image
            release_cuda_memory()

        logger.info("Total generation takes %.2fs", time.time() - start_time)
        return final_save_path, uid

    def close(self):
        with self._gpu_lock:
            self._unload_shape()
            self._unload_paint()

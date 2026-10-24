import os
import sys
import subprocess
import threading
import webbrowser
import io
import base64
import numpy as np
from PIL import Image

# Check for tkinter (standard on Windows/Mac, optional on Linux)
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except Exception:
    HAS_TKINTER = False

# Define the exact packages needed to run ModelScope and the GUI
REQUIRED_PACKAGES = {
    "diffusers": "diffusers",
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "nicegui": "nicegui",
    "pillow": "pillow"
}

def setup_independent_environment():
    """Checks your system for each dependency and installs any missing ones."""
    print("====================================================")
    print("        CHECKING SYSTEM FOR MODELSCOPE DEPENDENCIES ")
    print("====================================================")
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
            print(f"[✓] {pip_name} is ready.")
        except ImportError:
            print(f"[X] {pip_name} is MISSING. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"[✓] Successfully installed {pip_name}")
            except Exception as e:
                print(f"[!] Error installing {pip_name}: {str(e)}")
                sys.exit(1)

    print("\n====================================================")
    print("        LAUNCHING LOCAL INDEPENDENT INTERFACE       ")
    print("====================================================")

# Execute dependency check safely
setup_independent_environment()

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from nicegui import ui

pipe = None
is_generating = False

# Global state storage for uploaded image
uploaded_image_state = {"image": None, "b64": ""}

def load_pipeline(output_container):
    global pipe
    if pipe is None:
        with output_container:
            ui.notify("Downloading ModelScope weights directly from HuggingFace... (First launch only)", type="info")
        try:
            pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe = pipe.to("cuda")
            pipe.enable_model_cpu_offload()
            with output_container:
                ui.notify("ModelScope Pipeline Loaded Successfully!", type="positive")
        except Exception as e:
            with output_container:
                ui.notify(f"Engine Load Failed: {str(e)}", type="negative")

def generate_video_thread(mode, prompt_text, negative_prompt_text, input_image, strength_val,
                          frame_count, fps_count, width, height, output_container, progress_bar, progress_label):
    global is_generating
    matrix_notification = None
    try:
        load_pipeline(output_container)
        if pipe is None:
            is_generating = False
            return
        
        with output_container:
            matrix_notification = ui.notify(f"Processing Tensor matrix operations ({frame_count} frames)...", type="ongoing")
            progress_bar.set_value(0.0)
            progress_bar.visible = True
            progress_label.visible = True
        
        num_inference_steps = 50
        neg_prompt = negative_prompt_text.strip() if negative_prompt_text.strip() else None

        def classic_step_callback(step, timestep, latents):
            nonlocal matrix_notification
            try:
                progress_percentage = min(float(step + 1) / float(num_inference_steps), 1.0)
            except Exception:
                progress_percentage = 0.5
            with output_container:
                if matrix_notification:
                    matrix_notification.dismiss()
                    matrix_notification = None
                progress_bar.set_value(progress_percentage)
                progress_label.set_text(f"🎬 Denoising Progress: {int(progress_percentage * 100)}%")

        # Image-to-Video Mode
        if mode == 'Image to Video':
            active_prompt = prompt_text.strip() if prompt_text.strip() else "high quality motion video footage"
            
            # Prepare and encode image into latent space
            resized_img = input_image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
            img_arr = np.array(resized_img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float16)

            with torch.no_grad():
                # Encode via VAE
                init_latent = pipe.vae.encode(img_tensor).latent_dist.sample()
                init_latent = init_latent * pipe.vae.config.scaling_factor
                # Repeat across frames: [1, 4, num_frames, H//8, W//8]
                clean_latents = init_latent.unsqueeze(2).repeat(1, 1, int(frame_count), 1, 1)
                
                # Blend with Gaussian noise according to motion strength
                noise = torch.randn_like(clean_latents)
                s = float(strength_val)
                noisy_latents = ((1.0 - s) * clean_latents + s * noise)

            output = pipe(
                prompt=active_prompt,
                negative_prompt=neg_prompt,
                num_frames=int(frame_count),
                height=int(height),
                width=int(width),
                latents=noisy_latents,
                num_inference_steps=num_inference_steps,
                callback=classic_step_callback,
                callback_steps=1
            )
        # Text-to-Video Mode
        else:
            output = pipe(
                prompt=prompt_text,
                negative_prompt=neg_prompt,
                num_frames=int(frame_count),
                height=int(height),
                width=int(width),
                num_inference_steps=num_inference_steps,
                callback=classic_step_callback,
                callback_steps=1
            )
        
        raw_frames = output.frames
        video_frames = []

        if isinstance(raw_frames, list) and len(raw_frames) > 0 and isinstance(raw_frames[0], Image.Image):
            video_frames = raw_frames
        else:
            try:
                arr = np.array(raw_frames)
                arr = np.squeeze(arr)
                if arr.ndim == 3:
                    arr = np.expand_dims(arr, axis=0)
                if arr.ndim == 5:
                    arr = arr[0]
                    
                for frame in arr:
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    video_frames.append(Image.fromarray(frame))
            except Exception as matrix_err:
                print(f"[!] Matrix conversion fallback engaged: {str(matrix_err)}")
                video_frames = raw_frames

        output_filename = "modelscope_output.mp4"
        video_path = export_to_video(video_frames, fps=int(fps_count), output_video_path=output_filename)
        absolute_path = os.path.abspath(video_path)
        
        with output_container:
            progress_bar.visible = False
            progress_label.visible = False
            ui.notify("Success! Video compiled.", type="positive")
            output_container.clear()
            
            ui.markdown(f"### 🎉 Generation Complete!\nFile saved at:\n`{absolute_path}`")
            ui.video(absolute_path).classes('w-full max-w-xl rounded shadow-lg my-2')
            
    except Exception as e:
        with output_container:
            if matrix_notification:
                matrix_notification.dismiss()
            progress_bar.visible = False
            progress_label.visible = False
            ui.notify(f"Generation Failed: {str(e)}", type="negative")
    finally:
        is_generating = False

def start_generation(mode, prompt_text, negative_prompt_text, strength_val, frame_count, fps_count, width, height,
                     output_container, progress_bar, progress_label):
    global is_generating
    if is_generating:
        ui.notify("Another generation task is already processing!", type="warning")
        return
    
    if mode == 'Image to Video':
        if uploaded_image_state["image"] is None:
            ui.notify("Please choose an image first for Image-to-Video mode.", type="warning")
            return
    else:
        if not prompt_text.strip():
            ui.notify("Please enter a text prompt first.", type="warning")
            return
        
    is_generating = True
    output_container.clear()
    with output_container:
        ui.spinner(size='lg').classes('q-pa-md')
        ui.label("Initializing pipeline render loop...")
        
    threading.Thread(
        target=generate_video_thread, 
        args=(
            mode, 
            prompt_text, 
            negative_prompt_text, 
            uploaded_image_state["image"], 
            strength_val,
            frame_count, 
            fps_count, 
            width, 
            height, 
            output_container, 
            progress_bar, 
            progress_label
        ), 
        daemon=True
    ).start()

# --- GUI Layout Design ---
@ui.page('/')
def index():
    ui.query('body').style('background-color: #121212; color: #ffffff;')
    
    with ui.card().style('background-color: #1e1e1e; border: 1px solid #383838; color: #ffffff;').classes('w-full max-w-4xl mx-auto my-6 p-6 rounded-xl shadow-2xl font-sans'):
        # Requested Header at the very top of the localhost site
        ui.markdown("# 🍝 THE OG WILL SMITH EATING SPAGHETTI VIDEO GENERATOR SUITE").classes('text-center font-extrabold text-amber-400 tracking-wide text-2xl md:text-3xl my-2')
        ui.label("Render legendary meme clips offline using the classic ModelScope 1.7B engine.").classes('text-center text-sm text-gray-400 mb-6 block')
        
        # Generation Mode Switch
        with ui.row().classes('w-full items-center justify-center gap-6 my-2 p-2 bg-[#262626] rounded-lg border border-[#444]'):
            ui.label("🎬 Mode:").classes('font-bold text-gray-300')
            mode_radio = ui.radio(['Text to Video', 'Image to Video'], value='Text to Video').props('inline color="amber"')
        
        # Image-to-Video Container
        image_input_box = ui.column().classes('w-full p-4 my-2 rounded-lg bg-[#252525] border border-dashed border-[#555] items-center')
        image_input_box.bind_visibility_from(mode_radio, 'value', backward=lambda v: v == 'Image to Video')
        
        with image_input_box:
            ui.label("Choose Source Image for Image-to-Video").classes('text-sm font-semibold text-gray-300')
            
            upload_status_label = ui.label("No image loaded yet").classes('text-xs text-gray-400 italic my-1')
            preview_image = ui.image().classes('w-44 h-44 object-cover rounded-lg border border-[#666] my-2')
            preview_image.visible = False

            def set_loaded_pil_image(pil_img, filename="image"):
                uploaded_image_state["image"] = pil_img
                thumb = pil_img.copy()
                thumb.thumbnail((256, 256))
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=90)
                b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                uploaded_image_state["b64"] = f"data:image/jpeg;base64,{b64_str}"
                
                preview_image.set_source(uploaded_image_state["b64"])
                preview_image.visible = True
                clear_img_btn.visible = True
                upload_status_label.set_text(f"✓ Ready: {filename} ({pil_img.width}x{pil_img.height} px)")
                upload_status_label.classes(replace='text-xs text-green-400 font-semibold my-1')
                ui.notify("Image loaded successfully!", type="positive")

            # Native OS File Picker Dialog (100% reliable)
            def open_system_file_picker():
                if not HAS_TKINTER:
                    ui.notify("Tkinter not available. Please use the drag-and-drop or path box below.", type="warning")
                    return
                try:
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)
                    file_path = filedialog.askopenfilename(
                        title="Select Image File",
                        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp")]
                    )
                    root.destroy()
                    if file_path:
                        pil_img = Image.open(file_path).convert('RGB')
                        set_loaded_pil_image(pil_img, os.path.basename(file_path))
                except Exception as ex:
                    ui.notify(f"Could not open file: {ex}", type="negative")

            with ui.row().classes('gap-3 my-2 items-center'):
                ui.button('📁 Browse Image File...', on_click=open_system_file_picker).props('color="amber" text-color="black"')

            # Web upload event handler
            def handle_upload(e):
                try:
                    raw_bytes = None
                    for attr in ['content', 'file']:
                        if hasattr(e, attr):
                            val = getattr(e, attr)
                            if hasattr(val, 'read'):
                                raw_bytes = val.read()
                                break
                    if raw_bytes is None:
                        raise ValueError("Could not extract bytes from web upload. Use 'Browse Image File...' instead.")
                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
                    set_loaded_pil_image(pil_img, getattr(e, 'name', 'Uploaded File'))
                except Exception as err:
                    ui.notify(f"Upload notice: {str(err)}", type="negative")

            file_uploader = ui.upload(
                on_upload=handle_upload, 
                max_files=1, 
                auto_upload=True
            ).classes('max-w-xs my-1').props('accept="image/*" color="amber"')

            # Fallback path input
            with ui.row().classes('w-full max-w-md items-center justify-center gap-2 mt-2'):
                local_path_input = ui.input(placeholder='Or paste image path (C:/photo.jpg)').classes('flex-grow text-xs').props('dense outlined')
                
                def load_local_path():
                    path = local_path_input.value.strip().strip('"').strip("'")
                    if not path or not os.path.exists(path):
                        ui.notify("Path does not exist on disk.", type="warning")
                        return
                    try:
                        pil_img = Image.open(path).convert('RGB')
                        set_loaded_pil_image(pil_img, os.path.basename(path))
                    except Exception as ex:
                        ui.notify(f"Failed to load: {ex}", type="negative")
                        
                ui.button('Load Path', on_click=load_local_path).props('dense outline color="amber"').classes('text-xs')

            def clear_image():
                uploaded_image_state["image"] = None
                uploaded_image_state["b64"] = ""
                preview_image.visible = False
                preview_image.set_source('')
                clear_img_btn.visible = False
                file_uploader.reset()
                local_path_input.value = ''
                upload_status_label.set_text("No image loaded yet")
                upload_status_label.classes(replace='text-xs text-gray-400 italic my-1')

            clear_img_btn = ui.button('Remove Image', on_click=clear_image).props('flat color="red" dense').classes('mt-1')
            clear_img_btn.visible = False
            
            with ui.row().classes('w-full max-w-md items-center justify-between gap-4 mt-2'):
                strength_slider = ui.slider(min=0.2, max=0.95, value=0.70, step=0.05).classes('w-7/12')
                strength_label = ui.label().classes('text-xs text-gray-300 font-mono')
                strength_label.bind_text_from(strength_slider, 'value', backward=lambda v: f'Motion Intensity: {v:.2f}')
                
            ui.label("Higher = more transformation and motion; Lower = closer to original image.").classes('text-xs text-gray-500 italic')

        # Text Prompt
        prompt_input = ui.textarea(
            label='Text Prompt',
            placeholder='Will Smith enthusiastically eating a large bowl of spaghetti, cinematic meme footage...'
        ).style('background-color: #2b2b2b; color: #ffffff;').classes('w-full text-base rounded my-2')
        
        # Negative Prompt Input
        negative_prompt_input = ui.textarea(
            label='Negative Prompt (Optional)',
            placeholder='blurry, low quality, distorted, static, deformed...'
        ).style('background-color: #2b2b2b; color: #ffffff;').classes('w-full text-base rounded mt-2')
        
        # Requested Warning Label
        ui.label("⚠️ The negative prompt may not work, this is an old model after all.").classes('text-xs text-amber-400 font-semibold mb-4 ml-1')
        
        # Output Resolution Controls (Defaults to recommended 256x256)
        with ui.row().classes('w-full justify-between items-center gap-4 my-2 p-3 bg-[#262626] rounded border border-[#3a3a3a]'):
            with ui.column().classes('w-5/12'):
                width_label = ui.label().classes('text-xs font-bold text-gray-300')
                width_slider = ui.slider(min=128, max=512, value=256, step=64).classes('w-full')
                width_label.bind_text_from(
                    width_slider, 'value', 
                    backward=lambda v: f"↔ Width: {int(v)} px {'(Recommended)' if int(v) == 256 else ''}"
                )
            
            with ui.column().classes('w-5/12'):
                height_label = ui.label().classes('text-xs font-bold text-gray-300')
                height_slider = ui.slider(min=128, max=512, value=256, step=64).classes('w-full')
                height_label.bind_text_from(
                    height_slider, 'value', 
                    backward=lambda v: f"↕ Height: {int(v)} px {'(Recommended)' if int(v) == 256 else ''}"
                )
        
        # Frame Count & FPS Sliders
        with ui.row().classes('w-full justify-between items-center gap-4 my-2 p-3 bg-[#262626] rounded border border-[#3a3a3a]'):
            with ui.column().classes('w-5/12'):
                frame_label = ui.label().classes('text-xs font-bold text-gray-300')
                frame_slider = ui.slider(min=8, max=60, value=16, step=1).classes('w-full')
                frame_label.bind_text_from(frame_slider, 'value', backward=lambda v: f"🎬 Total Frames: {int(v)}")
            
            with ui.column().classes('w-5/12'):
                fps_label = ui.label().classes('text-xs font-bold text-gray-300')
                fps_slider = ui.slider(min=1, max=30, value=8, step=1).classes('w-full')
                fps_label.bind_text_from(fps_slider, 'value', backward=lambda v: f"⚡ Playback Speed: {int(v)} FPS")

        # Reset All Settings to Defaults
        def reset_all_defaults():
            mode_radio.value = 'Text to Video'
            prompt_input.value = ''
            negative_prompt_input.value = ''
            width_slider.value = 256
            height_slider.value = 256
            frame_slider.value = 16
            fps_slider.value = 8
            strength_slider.value = 0.70
            clear_image()
            ui.notify("All settings restored to defaults (256x256, 16 frames, 8 FPS).", type="info")

        with ui.row().classes('w-full justify-end my-2'):
            ui.button('↺ Reset All Settings to Defaults', on_click=reset_all_defaults).props('outline color="grey-5" dense').classes('text-xs')

        # Progress elements
        progress_label = ui.label("🎬 Denoising Progress: 0%").style('color: #f59e0b; font-weight: bold;').classes('w-full text-center text-sm my-1')
        progress_bar = ui.linear_progress(value=0.0, show_value=False).props('instant-feedback color="amber"').classes('w-full rounded my-2')
        
        progress_label.visible = False
        progress_bar.visible = False
        
        output_block = ui.column().classes('w-full items-center justify-center my-4')
        
        # Generation Button
        ui.button(
            'GENERATE VIDEO',
            on_click=lambda: start_generation(
                mode_radio.value,
                prompt_input.value,
                negative_prompt_input.value,
                strength_slider.value,
                frame_slider.value,
                fps_slider.value,
                width_slider.value,
                height_slider.value,
                output_block,
                progress_bar,
                progress_label
            )
        ).style('background-color: #f59e0b !important; color: #000000 !important; font-weight: 800; font-size: 1.25rem; letter-spacing: 0.05em;').classes('w-full py-4 rounded-lg shadow-xl hover:brightness-110')

# Open browser and run
threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8080")).start()
ui.run(port=8080, title="THE OG WILL SMITH EATING SPAGHETTI VIDEO GENERATOR SUITE", reload=False, show=False)
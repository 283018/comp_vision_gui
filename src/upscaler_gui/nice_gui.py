import logging
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import base64
import io
import traceback

import numpy as np
from nicegui import run, ui
from PIL import Image

try:
    from upscaler.image_api import Upscaler
except Exception:  # noqa: BLE001
    try:
        from src.upscaler.image_api import Upscaler
    except Exception as e:
        msg = "Application PATH is incorrect."
        raise RuntimeError(msg) from e

SCALE_CHOICES = [2, 4, 8, 16, 32]
DEFAULT_SCALE_INDEX = 1
DEFAULT_PATCH_SIZE = 128


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_thumbnail(img: Image.Image, max_size: int = 400) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail((max_size, max_size))
    return thumb


def prepare_image_for_upscaler(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    arr = np.ascontiguousarray(arr)
    return Image.fromarray(arr, mode="RGB")


def build_ui():  # noqa: PLR0915
    upscaler = Upscaler()
    upscaler.load()

    uploaded_image: Image.Image | None = None
    uploaded_name: str = ""
    result_container = None
    upload_card = None

    async def on_upload(e):
        nonlocal uploaded_image, uploaded_name, result_container, upload_card

        events = e if isinstance(e, list) else [e]

        for ev in events:
            try:
                data = None

                if hasattr(ev, "file") and ev.file is not None:
                    try:
                        data = await ev.file.read()
                    except TypeError:
                        data = ev.file.read()

                if data is None and hasattr(ev, "content"):
                    data = ev.content
                if data is None and hasattr(ev, "bytes"):
                    data = ev.bytes

                if not isinstance(data, (bytes, bytearray)):
                    continue

                img = Image.open(io.BytesIO(data)).convert("RGB")

                uploaded_image = img
                uploaded_name = (
                    getattr(ev, "filename", None)
                    or getattr(ev, "name", None)
                    or "upload.png"
                )

                if upload_card is not None:
                    upload_card.delete()
                    upload_card = None
                    result_container = None

                with ui.card().style("width: 720px; margin: 8px") as card:
                    upload_card = card  # <-- FIX
                    ui.label(f"Uploaded: {uploaded_name}")
                    ui.image(pil_to_data_url(make_thumbnail(img, 500)))
                    result_container = ui.column()

                ui.notify("Image uploaded", timeout=1.5)

            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                ui.notify(f"Upload handler error: {exc}", color="negative")

    async def on_upscale():
        nonlocal uploaded_image, uploaded_name, result_container

        if uploaded_image is None:
            ui.notify("No image uploaded", color="warning")
            return

        scale = SCALE_CHOICES[int(scale_slider.value)]
        overlap = int(overlap_slider.value)
        patch = int(patch_input.value)

        ui.notify("Starting upscaling...", timeout=2.0)

        safe_img = prepare_image_for_upscaler(uploaded_image)

        try:
            result = await run.io_bound(
                upscaler.upscale,
                safe_img,
                scale=scale,  # type: ignore
                patch_size=patch,
                overlap=overlap,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            ui.notify(f"Upscale failed: {exc}", color="negative")
            return

        try:
            data_url = pil_to_data_url(result)

            if result_container is not None:
                result_container.clear()
                with result_container:
                    ui.label("Result:")
                    ui.image(pil_to_data_url(make_thumbnail(result, 600)))
                    filename = f"upscaled_{scale}x_{uploaded_name}"

                    ui.html(
                        f'''
                        <a href="{data_url}" download="{filename}"
                           style="
                               background:#1976d2;
                               color:white;
                               padding:8px 14px;
                               border-radius:6px;
                               text-decoration:none;
                               display:inline-block;
                               font-weight:500;
                           ">
                           Save upscaled image
                        </a>
                        ''',
                        sanitize=False,
                    )
            else:
                ui.open(data_url, new_tab=True)  # type: ignore

            ui.notify("Upscaling finished", timeout=2.0)

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            ui.notify(f"Error presenting result: {exc}", color="negative")

    ui.label("Simple Image Upscaler").classes("text-h5")

    ui.upload(on_upload=on_upload, multiple=False).props("accept=image/*")

    with ui.row().style("gap: 24px; align-items:center"):
        scale_slider = ui.slider(
            min=0,
            max=len(SCALE_CHOICES) - 1,
            value=DEFAULT_SCALE_INDEX,
            step=1,
        ).style("width: 600px")
        scale_label = ui.label(
            f"Scale: {SCALE_CHOICES[DEFAULT_SCALE_INDEX]}x",
        ).style("min-width:160px")

        overlap_slider = ui.slider(min=0, max=256, value=0).style("width: 600px")
        overlap_label = ui.label("Overlap: 0 px").style("min-width:160px")

    def _update_scale_label(_=None):
        scale_label.set_text(f"Scale: {SCALE_CHOICES[int(scale_slider.value)]}x")

    def _update_overlap_label(_=None):
        overlap_label.set_text(f"Overlap: {int(overlap_slider.value)} px")

    scale_slider.on("update", _update_scale_label)
    scale_slider.on("input", _update_scale_label)
    scale_slider.on("change", _update_scale_label)

    overlap_slider.on("update", _update_overlap_label)
    overlap_slider.on("input", _update_overlap_label)
    overlap_slider.on("change", _update_overlap_label)

    _update_scale_label()
    _update_overlap_label()

    patch_input = ui.number(
        label="Patch size (px)",
        value=DEFAULT_PATCH_SIZE,
        min=16,
        max=4096,
        step=8,
    )

    ui.button("Upscale", on_click=on_upscale)


if __name__ in {"__main__", "__mp_main__"}:
    build_ui()
    ui.run(title="Upscaler", port=8080)

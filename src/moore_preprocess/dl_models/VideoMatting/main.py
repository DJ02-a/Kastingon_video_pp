import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from moore_preprocess import CKPT_DIR

from .inference_utils import (
    ImageSequenceReader,
    ImageSequenceWriter,
    VideoReader,
    VideoWriter,
)
from .model import MattingNetwork


class RobudstVideoMatting(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MattingNetwork().eval().cuda()  # or "resnet50"
        self.model.load_state_dict(
            torch.load(CKPT_DIR / "VideoMatting" / "rvm_mobilenetv3.pth")
        )

    def forward(
        self,
        input_source: str,
        input_resize: Optional[Tuple[int, int]] = None,
        downsample_ratio: Optional[float] = None,
        output_type="png_sequence",
        output_composition: Optional[str] = None,
        output_alpha: Optional[str] = None,
        output_foreground: Optional[str] = None,
        output_video_mbps: Optional[float] = None,
        seq_chunk: int = 1,
        num_workers: int = 0,
        progress: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
            input_resize: If provided, the input are first resized to (w, h).
            downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
            output_type: Options: ["video", "png_sequence"].
            output_composition:
                The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
                If output_type == 'video', the composition has green screen background.
                If output_type == 'png_sequence'. the composition is RGBA png images.
            output_alpha: The alpha output from the model.
            output_foreground: The foreground output from the model.
            seq_chunk: Number of frames to process at once. Increase it for better parallelism.
            num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
            progress: Show progress bar.
            device: Only need to manually provide if model is a TorchScript freezed model.
            dtype: Only need to manually provide if model is a TorchScript freezed model.
        """

        assert downsample_ratio is None or (
            downsample_ratio > 0 and downsample_ratio <= 1
        ), "Downsample ratio must be between 0 (exclusive) and 1 (inclusive)."
        assert any(
            [output_composition, output_alpha, output_foreground]
        ), "Must provide at least one output."
        assert output_type in [
            "video",
            "png_sequence",
        ], 'Only support "video" and "png_sequence" output modes.'
        assert seq_chunk >= 1, "Sequence chunk must be >= 1"
        assert num_workers >= 0, "Number of workers must be >= 0"

        # Initialize transform
        if input_resize is not None:
            transform = transforms.Compose(
                [transforms.Resize(input_resize[::-1]), transforms.ToTensor()]
            )
        else:
            transform = transforms.ToTensor()

        # Initialize reader
        if os.path.isfile(input_source):
            source = VideoReader(input_source, transform)
        else:
            source = ImageSequenceReader(input_source, transform)
        reader = DataLoader(
            source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers
        )

        # Initialize writers
        if output_type == "video":
            frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
            output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
            if output_composition is not None:
                writer_com = VideoWriter(
                    path=output_composition,
                    frame_rate=frame_rate,
                    bit_rate=int(output_video_mbps * 1000000),
                )
            if output_alpha is not None:
                writer_pha = VideoWriter(
                    path=output_alpha,
                    frame_rate=frame_rate,
                    bit_rate=int(output_video_mbps * 1000000),
                )
            if output_foreground is not None:
                writer_fgr = VideoWriter(
                    path=output_foreground,
                    frame_rate=frame_rate,
                    bit_rate=int(output_video_mbps * 1000000),
                )
        else:
            if output_composition is not None:
                writer_com = ImageSequenceWriter(output_composition, "png")
            if output_alpha is not None:
                writer_pha = ImageSequenceWriter(output_alpha, "png")
            if output_foreground is not None:
                writer_fgr = ImageSequenceWriter(output_foreground, "png")

        # Inference

        if device is None or dtype is None:
            param = next(self.model.parameters())
            dtype = param.dtype
            device = param.device

        if (output_composition is not None) and (output_type == "video"):
            bgr = (
                torch.tensor([120, 255, 155], device=device, dtype=dtype)
                .div(255)
                .view(1, 1, 3, 1, 1)
            )

        try:
            with torch.no_grad():
                bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
                rec = [None] * 4
                for src in reader:
                    if downsample_ratio is None:
                        downsample_ratio = self._auto_downsample_ratio(*src.shape[2:])

                    src = src.to(device, dtype, non_blocking=True).unsqueeze(
                        0
                    )  # [B, T, C, H, W]
                    fgr, pha, *rec = self.model(src, *rec, downsample_ratio)

                    if output_foreground is not None:
                        writer_fgr.write(fgr[0])
                    if output_alpha is not None:
                        writer_pha.write(pha[0])
                    if output_composition is not None:
                        if output_type == "video":
                            com = fgr * pha + bgr * (1 - pha)
                        else:
                            fgr = fgr * pha.gt(0)
                            com = torch.cat([fgr, pha], dim=-3)
                        writer_com.write(com[0])

                    bar.update(src.size(1))

        finally:
            # Clean up
            if output_composition is not None:
                writer_com.close()
            if output_alpha is not None:
                writer_pha.close()
            if output_foreground is not None:
                writer_fgr.close()

        return

    def _auto_downsample_ratio(self, h, w):
        """
        Automatically find a downsample ratio so that the largest side of the resolution be 512px.
        """
        return min(512 / max(h, w), 1)

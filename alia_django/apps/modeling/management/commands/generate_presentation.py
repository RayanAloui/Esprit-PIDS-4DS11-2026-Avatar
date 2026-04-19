from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.modeling.powerpoint_generation import (
    generate_presentation_for_product,
    generate_presentation_for_product_with_narrations,
)


class Command(BaseCommand):
    help = "Generate PowerPoint presentation (and optionally an avatar video) for one product"

    def add_arguments(self, parser):
        parser.add_argument(
            "--product",
            type=str,
            required=True,
            help="Product name from CSV",
        )
        parser.add_argument(
            "--csv",
            type=str,
            default=None,
            help="Optional CSV path",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Optional output folder",
        )
        parser.add_argument(
            "--video",
            action="store_true",
            default=False,
            help="Also generate the avatar video (.mp4) after the PPTX",
        )

    def handle(self, *args, **options):
        product_name = options["product"]
        csv_path     = Path(options["csv"])    if options["csv"]    else None
        output_dir   = Path(options["output"]) if options["output"] else None

        try:
            if options["video"]:
                self._generate_video(product_name, csv_path, output_dir)
            else:
                self._generate_pptx(product_name, csv_path, output_dir)
        except Exception as exc:
            raise CommandError(str(exc)) from exc

    # ── PPTX only ────────────────────────────────────────────────────────────

    def _generate_pptx(self, product_name, csv_path, output_dir):
        file_path = generate_presentation_for_product(
            product_name=product_name,
            csv_path=csv_path,
            output_dir=output_dir,
        )
        self.stdout.write(self.style.SUCCESS(
            f"Presentation created successfully: {file_path}"
        ))

    # ── PPTX + avatar video ───────────────────────────────────────────────────

    def _generate_video(self, product_name, csv_path, output_dir):
        try:
            from apps.modeling.video_generation import generate_avatar_video
            import asyncio
        except ImportError as e:
            raise CommandError(
                f"Video generation dependencies not installed: {e}\n"
                "Run: pip install moviepy pyrender trimesh[easy] pygltflib librosa soundfile"
            ) from e

        self.stdout.write(f"Generating PPTX for '{product_name}'...")
        pptx_path, narrations = generate_presentation_for_product_with_narrations(
            product_name=product_name,
            csv_path=csv_path,
            output_dir=output_dir,
        )
        self.stdout.write(self.style.SUCCESS(f"  PPTX: {pptx_path}"))

        self.stdout.write(f"Generating avatar video ({len(narrations)} slides)...")
        for i, text in enumerate(narrations, 1):
            preview = text[:80].replace("\n", " ")
            self.stdout.write(f"  Slide {i}: {preview}{'…' if len(text) > 80 else ''}")

        video_path = asyncio.run(
            generate_avatar_video(
                pptx_path=pptx_path,
                slide_texts=narrations,
                output_path=pptx_path.with_suffix(".mp4"),
            )
        )
        self.stdout.write(self.style.SUCCESS(f"  Video: {video_path}"))
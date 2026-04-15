from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.modeling.powerpoint_generation import (
    generate_presentation_for_product,
)


class Command(BaseCommand):
    help = "Generate PowerPoint presentation for one product"

    def add_arguments(self, parser):
        parser.add_argument(
            "--product",
            type=str,
            required=True,
            help="Product name from CSV"
        )

        parser.add_argument(
            "--csv",
            type=str,
            default=None,
            help="Optional CSV path"
        )

        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Optional output folder"
        )

    def handle(self, *args, **options):
        product_name = options["product"]
        csv_path = Path(options["csv"]) if options["csv"] else None
        output_dir = Path(options["output"]) if options["output"] else None

        try:
            file_path = generate_presentation_for_product(
                product_name=product_name,
                csv_path=csv_path,
                output_dir=output_dir
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"Presentation created successfully: {file_path}"
            )
        )
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.modeling.flyer_generation import generate_flyers_from_csv


class Command(BaseCommand):
    help = "Generate marketing flyers for products in a CSV."

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default=None,
            help="Path to CSV file (default: apps/modeling/data/vital_products.csv).",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output directory for generated flyer images.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Optional model/provider hint passed to the generator implementation.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Optional limit of products to process.",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"]) if options.get("csv") else None
        output_dir = Path(options["output"]) if options.get("output") else None
        model = options["model"]
        limit = options.get("limit")

        try:
            total = generate_flyers_from_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                model=model,
                limit=limit,
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS(f"Flyers generated successfully: {total}"))

import csv
import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class IntentConfig:
    """Configuration for an intent."""
    name: str
    templates: List[str]
    response_message: str
    output_filename: str


class LocationLoader:
    """Responsible for loading locations from CSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> List[str]:
        """Load locations from CSV file."""
        locations = []
        with open(self.filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location = row['LOCATION'].strip()
                if location:
                    locations.append(location)
        return locations


class PhraseGenerator:
    """Generates training phrases from templates and locations."""

    ENTITY_NAME = "locations"

    @classmethod
    def generate(cls, locations: List[str], templates: List[str]) -> List[Dict[str, Any]]:
        """Generate training phrases for all locations with given templates."""
        user_says = []

        for template in templates:
            user_says.extend(cls._generate_from_template(template, locations))

        return user_says

    @classmethod
    def _generate_from_template(cls, template: str, locations: List[str]) -> List[Dict[str, Any]]:
        """Generate phrases from a single template."""
        parts = template.split('{}')
        if len(parts) != 2:
            print(f"Warning: Invalid template format: {template}")
            return []

        prefix, suffix = parts

        return [
            cls._create_phrase_entry(prefix, location, suffix)
            for location in locations
        ]

    @classmethod
    def _create_phrase_entry(cls, prefix: str, location: str, suffix: str) -> Dict[str, Any]:
        """Create a single phrase entry with entity annotation."""
        return {
            "isTemplate": False,
            "count": 0,
            "updated": None,
            "data": [
                {"text": prefix, "userDefined": False},
                {
                    "text": location,
                    "alias": cls.ENTITY_NAME,
                    "meta": f"@{cls.ENTITY_NAME}",
                    "userDefined": True
                },
                {"text": suffix, "userDefined": False}
            ],
            "id": ""
        }


class IntentBuilder:
    """Builds Dialogflow intent JSON structure."""

    def __init__(self, intent_config: IntentConfig):
        self.config = intent_config

    def build(self, user_says: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build complete intent structure."""
        return {
            "name": self.config.name,
            "auto": True,
            "responses": [
                {
                    "messages": [
                        {
                            "type": "message",
                            "condition": "",
                            "speech": [self.config.response_message]
                        }
                    ],
                    "parameters": [
                        {
                            "name": "locations",
                            "dataType": "@locations",
                            "value": "$locations",
                            "isList": False
                        }
                    ]
                }
            ],
            "userSays": user_says
        }


class IntentFileWriter:
    """Handles writing intent data to JSON files."""

    @staticmethod
    def write(filepath: str, intent_data: Dict[str, Any]) -> None:
        """Write intent JSON to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(intent_data, f, indent=2, ensure_ascii=False)


class DialogflowLocationIntentGenerator:
    """Main orchestrator for generating start/end location intents."""

    START_TEMPLATES = [
        "Start from {}",
        "Begin at {}",
        "My first stop is {}",
        "The trip starts at {}",
        "I want to start at {}"
    ]

    END_TEMPLATES = [
        "End at {}",
        "Finish at {}",
        "My last stop is {}",
        "The trip ends at {}",
        "I want to end at {}"
    ]

    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.loader = LocationLoader(csv_filepath)
        self.writer = IntentFileWriter()

    def generate_all(self) -> None:
        """Generate all start and end location intent files."""
        locations = self.loader.load()

        intent_configs = [
            IntentConfig(
                name="set.start.location",
                templates=self.START_TEMPLATES,
                response_message="Got it. I've set the starting location.",
                output_filename="set.start.location.json"
            ),
            IntentConfig(
                name="set.end.location",
                templates=self.END_TEMPLATES,
                response_message="Alright. I've set the ending location.",
                output_filename="set.end.location.json"
            )
        ]

        for config in intent_configs:
            self._generate_intent(locations, config)

        print("✅ Generated: set.start.location.json and set.end.location.json")

    def _generate_intent(self, locations: List[str], config: IntentConfig) -> None:
        """Generate a single intent file."""
        user_says = PhraseGenerator.generate(locations, config.templates)
        builder = IntentBuilder(config)
        intent_data = builder.build(user_says)
        self.writer.write(config.output_filename, intent_data)


if __name__ == "__main__":
    generator = DialogflowLocationIntentGenerator("location_db_with_tags.csv")
    generator.generate_all()

import csv
import json
import random
from typing import List, Dict, Any, Set
from dataclasses import dataclass


@dataclass
class Location:
    """Represents a location with its associated tags."""
    name: str
    tags: List[str]


class LocationDataLoader:
    """Responsible for loading location data from CSV."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> tuple[List[Location], Set[str]]:
        """Load locations and extract all unique tags."""
        locations = []
        tags = set()

        with open(self.filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location_name = row['LOCATION'].strip()
                location_tags = [tag.strip() for tag in row['tags'].split(',')]

                locations.append(Location(location_name, location_tags))
                tags.update(location_tags)

        return locations, tags


class PhraseGenerator:
    """Generates training phrases from templates."""

    PHRASE_TEMPLATES = [
        "I want to find {}",
        "I'm looking for {}",
        "Can you show me {}?",
        "Where is {}?",
        "Find {} for me",
        "I want to look for {}"
    ]

    SAMPLE_RATIO = 0.1

    @classmethod
    def generate_from_sample(cls, entries: List[str], entity_name: str) -> List[Dict[str, Any]]:
        """Generate phrases from a sampled subset of entries."""
        sample_size = max(1, int(len(entries) * cls.SAMPLE_RATIO))
        sampled = random.sample(entries, sample_size)

        return [
            cls._create_phrase_entry(template.format(entry), entry, entity_name)
            for entry in sampled
            for template in cls.PHRASE_TEMPLATES
        ]

    @classmethod
    def generate_for_all(cls, entries: List[str], entity_name: str) -> List[Dict[str, Any]]:
        """Generate phrases ensuring all entries appear at least once."""
        return [
            cls._create_phrase_entry(template.format(entry), entry, entity_name)
            for entry in entries
            for template in cls.PHRASE_TEMPLATES
        ]

    @classmethod
    def _create_phrase_entry(cls, phrase: str, entity_text: str, entity_name: str) -> Dict[str, Any]:
        """Create a single phrase entry with entity annotation."""
        text_before = phrase.replace(entity_text, "")

        return {
            "isTemplate": False,
            "count": 0,
            "updated": None,
            "data": [
                {"text": text_before, "userDefined": False},
                {
                    "text": entity_text,
                    "alias": entity_name,
                    "meta": f"@{entity_name}",
                    "userDefined": True
                }
            ],
            "id": ""
        }


class IntentBuilder:
    """Builds Dialogflow intent structure."""

    def __init__(self, name: str, speech_response: str, parameter_config: Dict[str, Any]):
        self.name = name
        self.speech_response = speech_response
        self.parameter_config = parameter_config

    def build(self, user_says: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build complete intent JSON structure."""
        return {
            "name": self.name,
            "auto": True,
            "responses": [
                {
                    "messages": [
                        {
                            "type": "message",
                            "condition": "",
                            "speech": [self.speech_response]
                        }
                    ],
                    "parameters": [self.parameter_config]
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


class DialogflowFindIntentGenerator:
    """Main orchestrator for generating find location intents."""

    LOCATION_ENTITY = "locations"
    TAG_ENTITY = "tags"

    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.loader = LocationDataLoader(csv_filepath)
        self.writer = IntentFileWriter()

    def generate_all(self) -> None:
        """Generate all find intent files."""
        locations, tags = self.loader.load()

        # Generate location intent
        location_intent = self._create_location_intent(locations)
        self.writer.write("find.location.particular.json", location_intent)

        # Generate tag intent
        tag_intent = self._create_tag_intent(tags)
        self.writer.write("find.location.tags.json", tag_intent)

        print("✅ Intent files created successfully.")

    def _create_location_intent(self, locations: List[Location]) -> Dict[str, Any]:
        """Create intent for finding specific locations."""
        location_names = [loc.name for loc in locations]
        user_says = PhraseGenerator.generate_from_sample(location_names, self.LOCATION_ENTITY)

        parameter_config = {
            "name": "location",
            "dataType": f"@{self.LOCATION_ENTITY}",
            "value": "$location",
            "isList": False
        }

        builder = IntentBuilder(
            "find.location.particular",
            "Sure, here is what I found for that location.",
            parameter_config
        )

        return builder.build(user_says)

    def _create_tag_intent(self, tags: Set[str]) -> Dict[str, Any]:
        """Create intent for finding locations by tags."""
        user_says = PhraseGenerator.generate_for_all(list(tags), self.TAG_ENTITY)

        parameter_config = {
            "name": "tag",
            "dataType": f"@{self.TAG_ENTITY}",
            "value": "$tag",
            "isList": False
        }

        builder = IntentBuilder(
            "find.location.tags",
            "Here are some places matching that type.",
            parameter_config
        )

        return builder.build(user_says)


if __name__ == "__main__":
    generator = DialogflowFindIntentGenerator("location_db_with_tags.csv")
    generator.generate_all()

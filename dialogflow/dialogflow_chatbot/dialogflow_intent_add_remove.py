import csv
import json
import random
import itertools
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TrainingPhrase:
    """Represents a training phrase with its data."""
    text: str
    alias: str = ""
    meta: str = ""
    user_defined: bool = False


class LocationLoader:
    """Responsible for loading locations from CSV file."""

    @staticmethod
    def load_from_csv(filepath: str) -> List[str]:
        """Load locations from CSV file."""
        locations = []
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location = row['LOCATION'].strip()
                if location:
                    locations.append(location)
        return locations


def _make_entry(data: List[Dict]) -> Dict[str, Any]:
    """Create a training phrase entry."""
    return {
        "isTemplate": False,
        "count": 0,
        "updated": None,
        "data": data,
        "id": ""
    }


class PhraseGenerator:
    """Generates training phrases from templates and locations."""

    ENTITY_NAME = "locations"
    SAMPLE_RATIO = 0.1

    def __init__(self, locations: List[str]):
        self.locations = locations

    def generate(self, templates: List[str]) -> List[Dict[str, Any]]:
        """Generate training phrases from templates."""
        sample_size = max(1, int(len(self.locations) * self.SAMPLE_RATIO))
        sampled = random.sample(self.locations, sample_size)
        paired = list(itertools.combinations(sampled, 2))[:max(1, sample_size // 2)]

        phrases = []
        for template in templates:
            phrases.extend(self._process_template(template, sampled, paired))

        return phrases

    def _process_template(self, template: str, sampled: List[str],
                          paired: List[tuple]) -> List[Dict[str, Any]]:
        """Process a single template and generate phrases."""
        placeholder_count = template.count('{}')

        if placeholder_count not in (1, 2):
            print(f"Skipping invalid template: {template}")
            return []

        parts = template.split('{}')
        if len(parts) != placeholder_count + 1:
            print(f"Skipping invalid template split: {template}")
            return []

        if placeholder_count == 1:
            return [_make_entry(self._build_single(parts, loc))
                    for loc in sampled]

        return [_make_entry(self._build_pair(parts, a, b))
                for a, b in paired]

    def _build_single(self, parts: List[str], location: str) -> List[Dict]:
        """Build data for single location phrase."""
        return [
            {"text": parts[0], "userDefined": False},
            {"text": location, "alias": self.ENTITY_NAME,
             "meta": f"@{self.ENTITY_NAME}", "userDefined": True},
            {"text": parts[1], "userDefined": False},
        ]

    def _build_pair(self, parts: List[str], loc_a: str,
                    loc_b: str) -> List[Dict]:
        """Build data for paired location phrase."""
        return [
            {"text": parts[0], "userDefined": False},
            {"text": loc_a, "alias": self.ENTITY_NAME,
             "meta": f"@{self.ENTITY_NAME}", "userDefined": True},
            {"text": parts[1], "userDefined": False},
            {"text": loc_b, "alias": self.ENTITY_NAME,
             "meta": f"@{self.ENTITY_NAME}", "userDefined": True},
            {"text": parts[2], "userDefined": False},
        ]


class IntentBuilder:
    """Builds Dialogflow intent JSON structure."""

    def __init__(self, name: str, response_message: str):
        self.name = name
        self.response_message = response_message

    def build(self, user_says: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build complete intent structure."""
        return {
            "name": self.name,
            "auto": True,
            "responses": [
                {
                    "messages": [
                        {
                            "type": "message",
                            "condition": "",
                            "speech": [self.response_message]
                        }
                    ],
                    "parameters": [
                        {
                            "name": "location",
                            "dataType": "@locations",
                            "value": "$location",
                            "isList": True
                        }
                    ]
                }
            ],
            "userSays": user_says
        }


class IntentFileWriter:
    """Writes intent JSON files to disk."""

    @staticmethod
    def write(filepath: str, intent_data: Dict[str, Any]) -> None:
        """Write intent data to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(intent_data, f, indent=2, ensure_ascii=False)


class DialogflowIntentGenerator:
    """Main orchestrator for generating Dialogflow intents."""

    ADD_TEMPLATES = [
        "Add {}",
        "Can you add {}?",
        "I want to add {}",
        "Please include {}",
        "Put {} in my list",
        "Add {} and {}",
        "Please add both {} and {}",
        "Include {} and {} in my list",
        "I want to add {} as well as {}",
        "Add {} along with {}",
        "Put {} and {} in the list",
        "Could you add {} and also {}?",
    ]

    REMOVE_TEMPLATES = [
        "Remove {}",
        "Can you remove {}?",
        "I want to remove {}",
        "Please delete {}",
        "Remove {} and {}",
        "Can you remove {} and {}?",
        "Please delete {} and {}",
        "I want to remove both {} and {}",
    ]

    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.loader = LocationLoader()
        self.writer = IntentFileWriter()

    def generate_all(self) -> None:
        """Generate all intent files."""
        locations = self.loader.load_from_csv(self.csv_filepath)
        phrase_generator = PhraseGenerator(locations)

        # Generate add intent
        add_phrases = phrase_generator.generate(self.ADD_TEMPLATES)
        add_intent = IntentBuilder(
            "add.location",
            "Okay, I've added the location(s) to your list."
        ).build(add_phrases)

        # Generate remove intent
        remove_phrases = phrase_generator.generate(self.REMOVE_TEMPLATES)
        remove_intent = IntentBuilder(
            "remove.location",
            "Got it. I've removed the location(s)."
        ).build(remove_phrases)

        # Write files
        self._write_intent_files(add_intent, remove_intent)
        print("✅ Add and Remove intent files created successfully.")

    def _write_intent_files(self, add_intent: Dict, remove_intent: Dict) -> None:
        """Write all intent JSON files."""
        intent_files = [
            ("trip.create.add.location.json", add_intent),
            ("trip.create.remove.location.json", remove_intent),
            ("favourite.add.location.json", add_intent),
            ("favourite.remove.location.json", remove_intent),
        ]

        for filepath, intent_data in intent_files:
            self.writer.write(filepath, intent_data)


if __name__ == "__main__":
    generator = DialogflowIntentGenerator("location_db_with_tags.csv")
    generator.generate_all()

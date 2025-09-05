#!/usr/bin/env python3
"""
Synthetic Shakespearean Sonnet Generator
Generates high-quality synthetic sonnets for data augmentation.
"""

import re
import random
import time
import os
from dotenv import load_dotenv

load_dotenv()

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import requests
from openai import OpenAI


@dataclass
class SonnetValidation:
    is_valid: bool
    line_count: int
    rhyme_scheme: str
    syllable_counts: List[int]
    issues: List[str]


class SonnetValidator:
    """Validates structural and stylistic aspects of generated sonnets."""
    
    def __init__(self):
        # Common Early Modern English patterns
        self.archaic_patterns = [
            r'\b(thou|thee|thy|thine|hast|hath|dost|doth|art|tis|twas)\b',
            r'\b(ere|nay|yea|wherefore|whence|whilst|oft)\b',
            r"'(gainst|tis|twas|neath|mongst|cross|tween)",
            r'\w+(est|eth)\b'  # archaic verb endings
        ]
        
        # Simple rhyme detection (last stressed syllable)
        self.rhyme_endings = {}
    
    def count_syllables(self, line: str) -> int:
        """Approximate syllable count for iambic pentameter validation."""
        # Remove punctuation and convert to lowercase
        clean_line = re.sub(r'[^\w\s]', '', line.lower())
        
        # Simple syllable counting heuristic
        vowel_groups = re.findall(r'[aeiouy]+', clean_line)
        syllable_count = len(vowel_groups)
        
        # Adjust for common patterns
        if clean_line.endswith('ed') and not clean_line.endswith(('ted', 'ded')):
            syllable_count -= 1
        if clean_line.endswith('le') and len(clean_line) > 2:
            syllable_count += 1
            
        return max(1, syllable_count)  # At least 1 syllable per line
    
    def extract_rhyme_sound(self, line: str) -> str:
        """Extract approximate rhyme sound from line ending."""
        # Get last word, remove punctuation
        words = re.findall(r'\b\w+\b', line.lower())
        if not words:
            return ""
        
        last_word = words[-1]
        
        # Simple rhyme sound extraction (last 2-3 characters)
        if len(last_word) >= 3:
            return last_word[-3:]
        return last_word
    
    def check_rhyme_scheme(self, lines: List[str]) -> str:
        """Check if sonnet follows ABAB CDCD EFEF GG pattern."""
        if len(lines) != 14:
            return "Invalid"
        
        rhyme_sounds = [self.extract_rhyme_sound(line) for line in lines]
        
        # Shakespearean sonnet pattern: ABAB CDCD EFEF GG
        expected_pattern = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 6]  # ABAB CDCD EFEF GG
        
        # Map similar sounds
        sound_groups = {}
        group_id = 0
        actual_pattern = []
        
        for sound in rhyme_sounds:
            # Find similar existing sound
            matched_group = None
            for existing_sound, existing_group in sound_groups.items():
                if self.sounds_rhyme(sound, existing_sound):
                    matched_group = existing_group
                    break
            
            if matched_group is not None:
                actual_pattern.append(matched_group)
            else:
                sound_groups[sound] = group_id
                actual_pattern.append(group_id)
                group_id += 1
        
        # Check if pattern matches Shakespearean structure
        if actual_pattern == expected_pattern:
            return "ABAB CDCD EFEF GG"
        else:
            return "Non-standard"
    
    def sounds_rhyme(self, sound1: str, sound2: str) -> bool:
        """Simple rhyme detection."""
        if len(sound1) < 2 or len(sound2) < 2:
            return sound1 == sound2
        
        # Check suffix similarity
        return sound1[-2:] == sound2[-2:]
    
    def has_archaic_language(self, text: str) -> bool:
        """Check if text contains Early Modern English patterns."""
        for pattern in self.archaic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def validate_sonnet(self, sonnet_text: str) -> SonnetValidation:
        """Comprehensive sonnet validation."""
        lines = [line.strip() for line in sonnet_text.strip().split('\n') if line.strip()]
        issues = []
        
        # Line count check
        if len(lines) != 14:
            issues.append(f"Invalid line count: {len(lines)} (should be 14)")
        
        # Syllable count check (iambic pentameter ~ 10 syllables)
        syllable_counts = [self.count_syllables(line) for line in lines]
        for i, count in enumerate(syllable_counts):
            if count < 8 or count > 12:  # Allow some flexibility
                issues.append(f"Line {i+1} has {count} syllables (should be ~10)")
        
        # Rhyme scheme check
        rhyme_scheme = self.check_rhyme_scheme(lines)
        if rhyme_scheme != "ABAB CDCD EFEF GG":
            issues.append(f"Rhyme scheme is {rhyme_scheme} (should be ABAB CDCD EFEF GG)")
        
        # Archaic language check
        if not self.has_archaic_language(sonnet_text):
            issues.append("Missing Early Modern English vocabulary")
        
        is_valid = len(issues) <= 2  # Allow minor issues
        
        return SonnetValidation(
            is_valid=is_valid,
            line_count=len(lines),
            rhyme_scheme=rhyme_scheme,
            syllable_counts=syllable_counts,
            issues=issues
        )


class SonnetGenerator:
    """Generates synthetic Shakespearean sonnets using LLM APIs."""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.client = None

        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "https://api.openai.com/v1"
        
        if model:
            self.model = model
        else:
            self.model = "gpt-4"

        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        else:
            raise Exception("No API key provided for OpenAI. Please set the OPENAI_API_KEY environment variable.")

        self.validator = SonnetValidator()
        self.thematic_seeds = self.load_thematic_seeds()
        
    def load_thematic_seeds(self) -> List[Dict[str, str]]:
        """Load thematic seeds for diverse sonnet generation."""
        return [
            # Time & Mortality Themes
            {"theme": "time_passage", "concept": "the relentless march of time", "mood": "contemplative"},
            {"theme": "youth_aging", "concept": "the transition from youth to old age", "mood": "melancholic"},
            {"theme": "seasons", "concept": "autumn's approach and winter's cold", "mood": "reflective"},
            {"theme": "mortality", "concept": "death's inevitable embrace", "mood": "solemn"},
            {"theme": "legacy", "concept": "what remains after we are gone", "mood": "hopeful"},
            
            # Beauty & Nature Themes  
            {"theme": "natural_beauty", "concept": "a perfect spring morning", "mood": "joyful"},
            {"theme": "fading_beauty", "concept": "beauty's impermanence like flowers", "mood": "wistful"},
            {"theme": "celestial", "concept": "stars and moon as eternal witnesses", "mood": "romantic"},
            {"theme": "garden", "concept": "a secret garden blooming in solitude", "mood": "peaceful"},
            {"theme": "storm", "concept": "nature's fury and calm aftermath", "mood": "dramatic"},
            
            # Love & Relationships
            {"theme": "unrequited_love", "concept": "loving someone who cannot return affection", "mood": "yearning"},
            {"theme": "eternal_love", "concept": "love that transcends time and death", "mood": "passionate"},
            {"theme": "forbidden_love", "concept": "attraction that society condemns", "mood": "conflicted"},
            {"theme": "lost_love", "concept": "remembering a love now gone", "mood": "sorrowful"},
            {"theme": "new_love", "concept": "the joy of discovering mutual affection", "mood": "ecstatic"},
            
            # Art & Immortality
            {"theme": "poetry_power", "concept": "verse as a means to immortality", "mood": "confident"},
            {"theme": "artistic_creation", "concept": "the painter capturing eternal beauty", "mood": "inspired"},
            {"theme": "music", "concept": "harmony that moves the soul", "mood": "transcendent"},
            {"theme": "memory", "concept": "preserving precious moments in words", "mood": "tender"},
            {"theme": "truth", "concept": "art revealing deeper truths", "mood": "philosophical"},
            
            # Self & Identity
            {"theme": "self_doubt", "concept": "questioning one's worth and purpose", "mood": "uncertain"},
            {"theme": "transformation", "concept": "becoming someone new through experience", "mood": "determined"},
            {"theme": "solitude", "concept": "finding peace in aloneness", "mood": "serene"},
            {"theme": "identity", "concept": "discovering who one truly is", "mood": "searching"},
            {"theme": "wisdom", "concept": "lessons learned through life's trials", "mood": "wise"}
        ]
    
    def create_generation_prompt(self, seed: Dict[str, str]) -> str:
        """Create a detailed prompt for sonnet generation."""
        
        base_prompt = f"""Write a Shakespearean sonnet with the following specifications:

THEME: {seed['concept']}
MOOD: {seed['mood']}

STRUCTURAL REQUIREMENTS:
- Exactly 14 lines
- Rhyme scheme: ABAB CDCD EFEF GG (3 quatrains + 1 couplet)
- Iambic pentameter (approximately 10 syllables per line)
- Final couplet should provide resolution or thematic conclusion

LANGUAGE REQUIREMENTS:
- Use Early Modern English vocabulary: thou, thee, thy, thine, hast, hath, dost, doth, art
- Include archaic constructions: 'tis, 'ere, 'gainst, whilst, oft
- Use poetic inversions for meter: "Fair youth" instead of "young person"
- Include contractions: o'er, ne'er, e'en

LITERARY DEVICES:
- Include at least one metaphor related to nature
- Use personification (Time, Death, Beauty, etc.)
- Employ alliteration in at least two lines
- Create vivid imagery through concrete details

CONTENT GUIDANCE:
- Focus on {seed['concept']} throughout the sonnet
- Maintain {seed['mood']} emotional tone
- Ensure thematic unity across all quatrains
- Make the final couplet memorable and conclusive

Generate the complete sonnet now:"""

        return base_prompt
    
    def generate_sonnet_openai(self, seed: Dict[str, str]) -> Optional[str]:
        """Generate a sonnet using OpenAI API."""
        if not self.client:
            return None
            
        prompt = self.create_generation_prompt(seed)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in Shakespearean poetry and Early Modern English. Generate authentic-sounding Shakespearean sonnets that would be indistinguishable from the original works."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
    
    def generate_sonnet_local_fallback(self, seed: Dict[str, str]) -> str:
        """Fallback local generation using templates (for demo purposes)."""
        templates = [
            # Template 1: Time theme
            """When Time doth steal the roses from thy cheek,
And silver threads do crown thy golden hair,
What comfort then for weary hearts that seek
The beauty that no longer lingers there?
Yet in these lines thy loveliness shall dwell,
Though seasons change and years may come and go,
For poetry hath power to weave a spell
That keeps thee young while mortal bodies grow
Old and decay beneath Time's cruel hand,
Which spares no face however fair it be.
But thou shalt bloom eternal in this band
Of verses wrought with love and artistry.
  So long as men can read and hearts can feel,
  Thy beauty lives, which Time can never steal.""",
            
            # Template 2: Love theme  
            """How doth thy beauty set my heart aflame
With passion that no words can e'er express!
Thy gentle grace doth put the sun to shame,
Thy virtues do my weary soul caress.
When thou art near, the very air grows sweet
With fragrance of thy pure and noble mind,
And Time himself doth pause his quick retreat
To gaze upon perfection so refined.
Yet fear I that this love of mine may be
Too bold a venture for so base a heart,
That thou, so fair, couldst never stoop to see
In me a worthy player for love's part.
  But if thou wilt accept this humble verse,
  My joy shall be beyond what words rehearse."""
        ]
        
        # Simple template selection based on theme
        if 'time' in seed['theme'] or 'mortality' in seed['theme']:
            return templates[0]
        else:
            return templates[1]
    
    def generate_sonnet(self, seed: Dict[str, str]) -> Optional[str]:
        """Generate a sonnet using available method."""
        # Try OpenAI API first
        sonnet = self.generate_sonnet_openai(seed)
        
        # Fallback to local generation if API fails
        if not sonnet:
            print(f"API generation failed, using fallback for theme: {seed['theme']}")
            sonnet = self.generate_sonnet_local_fallback(seed)
        
        return sonnet
    
    def generate_batch(self, count: int, validate: bool = True) -> List[str]:
        """Generate a batch of sonnets with optional validation."""
        sonnets = []
        attempts = 0
        max_attempts = count * 4  # Allow some failed attempts
        
        while len(sonnets) < count and attempts < max_attempts:
            # Select random thematic seed
            seed = random.choice(self.thematic_seeds)
            
            sonnet = self.generate_sonnet(seed)
            if not sonnet:
                attempts += 1
                continue
            
            # Validate if requested
            if validate:
                validation = self.validator.validate_sonnet(sonnet)
                if validation.is_valid:
                    sonnets.append(sonnet)
                    print(f"Generated valid sonnet {len(sonnets)}/{count} (theme: {seed['theme']})")
                else:
                    print(f"Invalid sonnet rejected (theme: {seed['theme']}): {validation.issues}")
            else:
                sonnets.append(sonnet)
                print(f"Generated sonnet {len(sonnets)}/{count} (theme: {seed['theme']})")
            
            attempts += 1
            
            # Rate limiting for API calls
            if self.client:
                time.sleep(1)  # 1 second between requests
        
        return sonnets


class SonnetDatasetFormatter:
    """Formats generated sonnets to match original dataset structure."""
    
    @staticmethod
    def format_sonnets_for_dataset(sonnets: List[str], start_number: int = 1000) -> str:
        """Format sonnets to match the original sonnets.txt structure."""
        formatted_output = []
        
        for i, sonnet in enumerate(sonnets):
            # Add sonnet number
            sonnet_number = start_number + i
            formatted_output.append(f"{sonnet_number}")
            formatted_output.append("")  # Empty line after number
            
            # Add sonnet lines
            lines = [line.strip() for line in sonnet.strip().split('\n') if line.strip()]
            formatted_output.extend(lines)
            formatted_output.append("")  # Empty line after sonnet
        
        return '\n'.join(formatted_output)
    
    @staticmethod
    def save_augmented_dataset(sonnets: List[str], filename: str = "data/sonnets_augmented.txt"):
        """Save sonnets in the same format as original dataset."""
        
        # Read original header if it exists
        header = """Sonnets
by William Shakespeare
Edited by Barbara A. Mowat and Paul Werstine
  with Michael Poston and Rebecca Niles
Folger Shakespeare Library
https://shakespeare.folger.edu/shakespeares-works/shakespeares-sonnets/
Created on Jul 31, 2015, from FDT version 0.9.0.1




"""
        
        # Format new sonnets
        formatted_content = SonnetDatasetFormatter.format_sonnets_for_dataset(sonnets)
        
        # Combine header and content
        full_content = header + formatted_content
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        print(f"Saved {len(sonnets)} synthetic sonnets to {filename}")


def main():
    """Main function to generate synthetic sonnets."""
    print("🌹 Shakespearean Sonnet Generator 🌹")
    print("=" * 50)
    
    # Configuration
    TARGET_COUNT = 500  # Adjust as needed (500-1000)
    BATCH_SIZE = 50
    VALIDATE = True
    
    # Try to get OpenAI API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise Exception("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
    else:
        print("✅ OpenAI API key found.")
    
    # Initialize generator
    base_url = "https://api.deepseek.com"
    model = "deepseek-reasoner"
    generator = SonnetGenerator(api_key=api_key, base_url=base_url, model=model)
    
    # Generate sonnets in batches
    all_sonnets = []
    batches = (TARGET_COUNT + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    for batch_num in range(batches):
        remaining = TARGET_COUNT - len(all_sonnets)
        batch_count = min(BATCH_SIZE, remaining)
        
        print(f"\n📝 Generating batch {batch_num + 1}/{batches} ({batch_count} sonnets)...")
        
        batch_sonnets = generator.generate_batch(batch_count, validate=VALIDATE)
        all_sonnets.extend(batch_sonnets)
        
        print(f"✅ Batch complete. Total sonnets: {len(all_sonnets)}/{TARGET_COUNT}")
    
    # Save results
    if all_sonnets:
        print(f"\n💾 Saving {len(all_sonnets)} sonnets to data/sonnets_augmented.txt...")
        SonnetDatasetFormatter.save_augmented_dataset(all_sonnets)
        print("✅ Generation complete!")
        
        # Print sample sonnet
        if all_sonnets:
            print("\n📖 Sample generated sonnet:")
            print("-" * 30)
            print(all_sonnets[0])
            print("-" * 30)
    else:
        print("❌ No valid sonnets were generated.")


if __name__ == "__main__":
    main()
import os
import asyncio
import datetime
import re
import sys
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import Channel, Message
from telethon.tl.functions.channels import GetForumTopicsRequest
from openai import OpenAI
from TTS.api import TTS  # Coqui TTS
import langdetect  # For language detection
from gtts import gTTS  # Google Text-to-Speech as a fallback
from pydub import AudioSegment
import random
import glob
import numpy as np

# Load environment variables from .env file
load_dotenv()

# ---------------------------
# Configuration and Setup
# ---------------------------

# Telegram configuration (set these as environment variables or replace directly)
API_ID = os.environ.get("TELEGRAM_API_ID")         # e.g., "123456"
API_HASH = os.environ.get("TELEGRAM_API_HASH")       # e.g., "abcdef123456..."
PHONE_NUMBER = os.environ.get("TELEGRAM_PHONE_NUMBER")  # e.g., "+1234567890"
SESSION_NAME = os.path.join("sessions", "telegram_session")

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OpenAI configuration
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Default settings
DEFAULT_MESSAGE_LIMIT = 1000

# ---------------------------
# Functions
# ---------------------------

def detect_language(text: str) -> str:
    """
    Detects the dominant language in the provided text.
    Returns the language code (e.g., 'en', 'ru').
    """
    try:
        # Use a sample of text to determine language (for efficiency)
        text_sample = text[:1000]
        return langdetect.detect(text_sample)
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"  # Default to English if detection fails

def generate_summary(chat_text: str, language: str = "en", topic_name: str = None) -> str:
    """
    Generates a podcast-style summary using OpenAI's GPT-3.5-turbo.
    The prompt instructs the model to produce a script that would last 5-10 minutes.
    Includes information about participants and their main points/theses.
    
    If topic_name is provided, it will be included in the prompt to provide context.
    """
    # Adjust system prompt based on detected language and whether there's a topic
    if language == "ru":
        topic_context = f" по теме '{topic_name}'" if topic_name else ""
        system_prompt = (
            f"Вы - эксперт по написанию подкастов. Обобщите следующую чат-беседу{topic_context} "
            "в четкий, увлекательный сценарий подкаста, который будет длиться около 5-10 минут при "
            "чтении вслух. Сценарий должен быть разговорным и естественным. "
            "Обязательно включите следующие детали:\n"
            "1. Имена участников чата (кто отправлял сообщения)\n"
            "2. Кто участвовал в обсуждении\n"
            "3. Основные тезисы и ключевые точки дискуссии от каждого участника\n"
            "Пожалуйста, создайте сценарий на русском языке."
        )
    else:
        topic_context = f" on the topic '{topic_name}'" if topic_name else ""
        system_prompt = (
            f"You are an expert podcast script writer. Summarize the following chat conversation{topic_context} "
            "into a clear, engaging podcast script that would last about 5-10 minutes when read aloud. "
            "The script should be conversational and natural. "
            "Be sure to include these specific details:\n"
            "1. Names of the participants (who sent messages)\n"
            "2. Who participated in the discussion\n"
            "3. The main theses and key points from each participant\n"
        )
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_text},
        ],
        max_tokens=1500  # Adjust this based on desired summary length
    )
    
    return response.choices[0].message.content

def generate_multi_topic_summary(topics_summaries: dict, language: str = "en") -> str:
    """
    Generates a consolidated summary for multiple topics.
    Takes a dictionary of {topic_name: summary} and creates a cohesive podcast script.
    """
    # Create a prompt for generating a consolidated summary
    if language == "ru":
        system_prompt = (
            "Вы - эксперт по написанию подкастов. Перед вами несколько кратких обзоров "
            "по разным темам из одного чата. Объедините их в единый согласованный сценарий подкаста, "
            "который представляет все эти темы в логической последовательности. "
            "Сценарий должен иметь четкое введение, представляющее общий контекст чата, "
            "а затем переходить к каждой теме, прежде чем завершиться кратким заключением. "
            "Сохраните ключевые детали и основные моменты из каждого обзора темы.\n"
            "Пожалуйста, создайте сценарий на русском языке."
        )
    else:
        system_prompt = (
            "You are an expert podcast script writer. You are given several summaries "
            "for different topics from the same chat. Combine them into a single cohesive podcast script "
            "that presents all these topics in a logical sequence. "
            "The script should have a clear introduction presenting the overall context of the chat, "
            "then transition to each topic, before concluding with a brief wrap-up. "
            "Preserve the key details and main points from each topic summary.\n"
        )
    
    # Build the content with topic summaries
    content = "Here are the summaries for each topic:\n\n"
    for topic, summary in topics_summaries.items():
        content += f"TOPIC: {topic}\n{summary}\n\n"
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=2000  # Increased token limit for the combined summary
    )
    
    return response.choices[0].message.content

def synthesize_speech_with_coqui(text: str, language: str = "en", output_filename: str = None) -> str:
    """
    Converts the given text to natural-sounding speech using multiple voices for different speakers.
    Uses Coqui TTS with different speaker IDs for different characters.
    Properly handles punctuation to make speech sound more natural.
    """
    if output_filename is None:
        output_filename = os.path.join(OUTPUT_DIR, "podcast_summary.wav")
    else:
        output_filename = os.path.join(OUTPUT_DIR, output_filename)
        
    try:
        # Extract different speakers and their lines from the text
        # Basic pattern: "Speaker Name: Their dialogue"
        speaker_pattern = re.compile(r'([^:]+):\s*(.*?)(?=\n[^:]+:|$)', re.DOTALL)
        segments = speaker_pattern.findall(text)
        
        if not segments:
            # If no clear speaker segments found, process as a single voice
            print("No distinct speakers found, using single voice")
            segments = [("Narrator", text)]
            
        # First make sure we have a reference voice file
        ref_voice_path = os.path.join("speakers", "voice1.wav")
        if not os.path.exists(ref_voice_path):
            # Create a directory for speakers
            os.makedirs("speakers", exist_ok=True)
            print(f"WARNING: Reference voice file not found at {ref_voice_path}")
            print("Falling back to Google TTS...")
            return synthesize_with_google_tts(segments, language, output_filename)
        else:
            print(f"Found reference voice file: {ref_voice_path}")
        
        # Load appropriate TTS model based on language
        if language == "ru":
            try:
                # First attempt with a multi-language model that supports Russian
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
                print("Using multilingual XTTS model for Russian")
            except Exception as e:
                print(f"Couldn't load multilingual XTTS model: {e}")
                try:
                    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True, gpu=False)
                    print("Using YourTTS model for Russian")
                except Exception as e:
                    print(f"Couldn't load YourTTS model: {e}")
                    # Fallback to Google TTS for Russian
                    return synthesize_with_google_tts(segments, language, output_filename)
        else:
            # For English, try to use a model with multiple speaker support
            try:
                # Try XTTS for English too as it has better multi-speaker support
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
                print("Using XTTS model for English with multiple speakers")
            except Exception as e:
                print(f"Couldn't load XTTS model: {e}")
                try:
                    # Try another model with speaker diversity
                    tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True, gpu=False)
                    print("Using VCTK model with multiple speakers")
                except Exception as e:
                    print(f"Couldn't load VCTK model: {e}")
                    try:
                        # Fallback to basic model
                        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
                        print("Using LJSpeech model (limited voice variety)")
                    except Exception as e:
                        print(f"Couldn't load any Coqui TTS model: {e}")
                        # Final fallback to Google TTS
                        return synthesize_with_google_tts(segments, language, output_filename)
        
        # Process each segment with a different voice
        audio_segments = []
        
        for i, (speaker, line) in enumerate(segments):
            speaker_name = speaker.strip()
            
            # Clean the text to avoid reading punctuation
            clean_text = clean_text_for_tts(line)
            if not clean_text.strip():
                continue
            
            # Create a unique output file for this segment
            segment_filename = f"segment_{i}.wav"
            
            # Check model type and use appropriate method
            if "xtts" in tts.model_name:
                try:
                    # Explicitly use the reference voice file for XTTS
                    print(f"Using reference voice: {ref_voice_path}")
                    tts.tts_to_file(
                        text=clean_text, 
                        file_path=segment_filename,
                        speaker_wav=ref_voice_path,
                        language=language[:2]  # Use first 2 chars of language code
                    )
                except Exception as e:
                    print(f"XTTS synthesis failed: {e}, skipping to next segment")
                    continue
            elif hasattr(tts, "speakers") and tts.speakers:
                # For models with built-in speaker support
                speaker_id = tts.speakers[i % len(tts.speakers)]
                try:
                    tts.tts_to_file(text=clean_text, file_path=segment_filename, speaker=speaker_id)
                except Exception as e:
                    print(f"Speaker-based TTS failed: {e}, trying default")
                    tts.tts_to_file(text=clean_text, file_path=segment_filename)
            else:
                # Basic TTS without speaker support
                tts.tts_to_file(text=clean_text, file_path=segment_filename)
            
            # Add to our collection of segments
            audio_segments.append(segment_filename)
        
        # Combine all audio segments into one file
        combined_audio = combine_audio_segments(audio_segments, output_filename)
        
        # Clean up temporary files
        for segment_file in audio_segments:
            if os.path.exists(segment_file):
                os.remove(segment_file)
        
        return combined_audio
    
    except Exception as e:
        print(f"Error synthesizing speech with Coqui TTS: {e}")
        print("Falling back to Google TTS...")
        try:
            return synthesize_with_google_tts(segments if 'segments' in locals() else [(None, text)], language, output_filename)
        except Exception as e2:
            print(f"Google TTS fallback also failed: {e2}")
            print("Unable to synthesize speech. Returning empty file path.")
            return ""

def synthesize_with_google_tts(segments, language, output_filename):
    """
    Fallback to Google TTS when Coqui fails.
    Attempts to use different voices for different speakers.
    """
    print("Using Google TTS for speech synthesis")
    
    # Google TTS uses MP3 format
    mp3_output = output_filename.replace(".wav", ".mp3")
    
    if len(segments) > 1:
        # Multiple speakers
        audio_segments = []
        
        for i, (speaker, line) in enumerate(segments):
            if not line.strip():
                continue
                
            # Clean the text to avoid reading punctuation
            clean_text = clean_text_for_tts(line)
            if not clean_text.strip():
                continue
            
            # Create a temporary file for this segment
            segment_file = os.path.join(OUTPUT_DIR, f"temp_segment_{i}.mp3")
            
            # Get language code for TTS
            lang_code = 'ru' if language == 'ru' else 'en'
            
            # Create a "different" voice by slightly adjusting speed/pitch
            # This is a simple approach since Google TTS has limited voice options
            tld = random.choice(['com', 'co.uk', 'com.au', 'ca', 'co.in']) if lang_code == 'en' else 'com'
            
            # Generate speech for this segment
            gtts = gTTS(text=clean_text, lang=lang_code, tld=tld, slow=False)
            gtts.save(segment_file)
            
            audio_segments.append(segment_file)
        
        # Combine all audio segments
        combined_audio = combine_audio_segments(audio_segments, mp3_output, format="mp3")
        
        # Clean up temporary files
        for segment_file in audio_segments:
            if os.path.exists(segment_file):
                os.remove(segment_file)
        
        return combined_audio
    else:
        # Single speaker case
        clean_text = clean_text_for_tts(segments[0][1]) if segments else text
        lang_code = 'ru' if language == 'ru' else 'en'
        gtts = gTTS(text=clean_text, lang=lang_code, slow=False)
        gtts.save(mp3_output)
        print(f"Speech synthesized using Google TTS and saved to {mp3_output}")
        return mp3_output

def clean_text_for_tts(text):
    """
    Clean text to improve TTS naturalness.
    - Replace punctuation with appropriate pauses
    - Format text to sound more natural when spoken
    """
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Add pauses for better speech rhythm
    text = re.sub(r'([.!?])\s+', r'\1\n', text)  # Add newline after sentence endings
    
    # Remove unnecessary characters that TTS may try to verbalize
    text = re.sub(r'["\'""'']', '', text)  # Remove quotes that might be read
    text = re.sub(r'[*_~]', '', text)  # Remove markdown characters
    
    # Convert dashes to commas for better phrasing
    text = re.sub(r'\s-\s', ', ', text)
    
    # Add subtle pauses for parenthetical content
    text = re.sub(r'\(', ', ', text)
    text = re.sub(r'\)', ', ', text)
    
    return text

def combine_audio_segments(segment_files, output_file, format="wav"):
    """
    Combines multiple audio segments into a single file.
    Adds short pauses between speaker segments for natural conversation flow.
    """
    combined = AudioSegment.empty()
    
    for i, segment_file in enumerate(segment_files):
        if not os.path.exists(segment_file):
            continue
            
        # Load the audio segment
        segment = AudioSegment.from_file(segment_file)
        
        # Add a short pause between speakers (but not before the first one)
        if i > 0:
            pause = AudioSegment.silent(duration=700)  # 700ms pause
            combined += pause
        
        # Add this segment to the combined audio
        combined += segment
    
    # Export the combined audio
    combined.export(output_file, format=format)
    return output_file

def parse_date(date_str: str) -> datetime.datetime:
    """
    Parse date string in various formats.
    Supports formats like: YYYY-MM-DD, DD/MM/YYYY, etc.
    """
    formats = [
        "%Y-%m-%d",        # 2023-01-31
        "%d/%m/%Y",        # 31/01/2023
        "%m/%d/%Y",        # 01/31/2023
        "%d-%m-%Y",        # 31-01-2023
        "%m-%d-%Y",        # 01-31-2023
        "%Y/%m/%d",        # 2023/01/31
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date: {date_str}. Please use format YYYY-MM-DD.")

async def check_if_forum(client: TelegramClient, entity) -> bool:
    """
    Check if a chat is a forum (has topics).
    """
    if not isinstance(entity, Channel):
        return False
    
    try:
        # Try to get forum topics - this will fail if it's not a forum
        topics = await client(GetForumTopicsRequest(
            channel=entity,
            offset_date=0,
            offset_id=0,
            offset_topic=0,
            limit=1
        ))
        return True
    except Exception:
        return False

async def get_forum_topics(client: TelegramClient, entity) -> list:
    """
    Retrieves the list of topics in a forum chat.
    """
    try:
        topics = await client(GetForumTopicsRequest(
            channel=entity,
            offset_date=0,
            offset_id=0,
            offset_topic=0,
            limit=100  # Fetch up to 100 topics
        ))
        return topics.topics
    except Exception as e:
        print(f"Error fetching forum topics: {e}")
        return []

async def fetch_topic_messages(client: TelegramClient, chat_id, topic_id, limit, start_date=None, end_date=None) -> list:
    """
    Fetches messages from a specific topic in a forum chat.
    """
    messages = []
    
    # For date filtering with topics
    if start_date or end_date:
        batch_size = min(100, limit)
        total_collected = 0
        offset_id = 0
        
        while total_collected < limit:
            batch = await client.get_messages(
                chat_id, 
                limit=batch_size, 
                offset_id=offset_id,
                reply_to=topic_id  # This filters by topic
            )
            
            if not batch:
                break  # No more messages
            
            for msg in batch:
                if msg.date is None:
                    continue
                    
                msg_datetime = msg.date.replace(tzinfo=None)  # Remove timezone for comparison
                
                # Check if message is within date range
                is_after_start = True if start_date is None else msg_datetime >= start_date
                is_before_end = True if end_date is None else msg_datetime <= end_date
                
                if is_after_start and is_before_end and msg.message:
                    messages.append(msg)
                    total_collected += 1
                    if total_collected >= limit:
                        break
            
            if len(batch) < batch_size:
                break  # No more messages to fetch
                
            offset_id = batch[-1].id
    else:
        # Simple case without date filtering
        messages = await client.get_messages(
            chat_id, 
            limit=limit,
            reply_to=topic_id  # This filters by topic
        )
    
    # Sort messages by date (oldest first)
    messages.sort(key=lambda msg: msg.date)
    return messages

def extract_message_text(messages: list) -> str:
    """
    Extracts text with sender information from a list of messages.
    """
    chat_lines = []
    for msg in messages:
        if msg.message:
            sender = msg.sender
            sender_name = getattr(sender, 'first_name', '') or ''
            if hasattr(sender, 'last_name') and sender.last_name:
                sender_name += ' ' + sender.last_name
            
            # If no name is available, try using username or id
            if not sender_name and hasattr(sender, 'username') and sender.username:
                sender_name = sender.username
            elif not sender_name:
                sender_name = f"User-{sender.id}" if hasattr(sender, 'id') else "Unknown"
            
            chat_lines.append(f"{sender_name}: {msg.message}")
    
    return "\n".join(chat_lines)

async def fetch_chat_history(client: TelegramClient) -> str:
    """
    Retrieves chat history from a selected chat.
    Lists available chats and allows user to specify message limit and date range.
    Now supports topics in forum chats.
    """
    dialogs = await client.get_dialogs()
    print("Available chats:")
    for i, dialog in enumerate(dialogs):
        print(f"{i}: {dialog.name}")

    selected_index = int(input("Enter the index of the chat to summarize: "))
    selected_dialog = dialogs[selected_index]
    chat_id = selected_dialog.id
    
    # Check if the selected chat is a forum (has topics)
    is_forum = await check_if_forum(client, selected_dialog.entity)
    
    # Ask for message limit
    limit_input = input(f"Enter the number of messages to fetch (default: {DEFAULT_MESSAGE_LIMIT}): ")
    limit = int(limit_input) if limit_input.strip() else DEFAULT_MESSAGE_LIMIT
    
    # Ask for date range
    use_date_range = input("Do you want to filter by date range? (y/n): ").lower().startswith('y')
    
    start_date = None
    end_date = None
    
    if use_date_range:
        try:
            start_date_str = input("Enter start date (YYYY-MM-DD, leave empty for no start date): ")
            if start_date_str.strip():
                start_date = parse_date(start_date_str)
                # Set to beginning of the day
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                
            end_date_str = input("Enter end date (YYYY-MM-DD, leave empty for today): ")
            if end_date_str.strip():
                end_date = parse_date(end_date_str)
                # Set to end of the day
                end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            else:
                end_date = datetime.datetime.now()
                
            print(f"Fetching messages from {start_date or 'the beginning'} to {end_date}")
        except ValueError as e:
            print(f"Error with dates: {e}")
            print("Proceeding without date filtering.")
            use_date_range = False
    
    # Handle differently based on whether it's a forum or regular chat
    if is_forum:
        print("This is a forum chat with topics!")
        topics = await get_forum_topics(client, selected_dialog.entity)
        
        if not topics:
            print("No topics found in this forum.")
            return ""
        
        print("Available topics:")
        for i, topic in enumerate(topics):
            # Skip the default "General" topic (0) in listing
            if topic.id != 0:
                print(f"{i}: {topic.title}")
        
        # Allow selecting all topics or specific ones
        topic_selection = input("Enter topic numbers to summarize (comma-separated), or 'all' for all topics: ")
        
        if topic_selection.lower() == 'all':
            selected_topics = [topic for topic in topics if topic.id != 0]  # Skip "General" topic
        else:
            topic_indices = [int(idx.strip()) for idx in topic_selection.split(',')]
            selected_topics = [topics[idx] for idx in topic_indices]
        
        topic_summaries = {}
        
        for topic in selected_topics:
            print(f"Fetching messages for topic: {topic.title}")
            
            # Fetch messages for this topic
            topic_messages = await fetch_topic_messages(
                client, chat_id, topic.id, limit, start_date, end_date
            )
            
            if not topic_messages:
                print(f"No messages found in topic '{topic.title}'")
                continue
            
            # Extract text from messages
            topic_text = extract_message_text(topic_messages)
            
            if not topic_text.strip():
                print(f"No text content found in topic '{topic.title}'")
                continue
                
            topic_summaries[topic.title] = topic_text
            print(f"Fetched {len(topic_messages)} messages from topic '{topic.title}'")
        
        if not topic_summaries:
            print("No messages found in any of the selected topics.")
            return ""
            
        return topic_summaries
    else:
        # Regular chat (no topics) - use existing logic
        messages = []
        
        if use_date_range:
            # When using date range, we might need to fetch more messages to get enough within the range
            # We'll fetch in batches to avoid hitting limits
            batch_size = min(100, limit)
            total_collected = 0
            offset_id = 0
            
            while total_collected < limit:
                batch = await client.get_messages(chat_id, limit=batch_size, offset_id=offset_id)
                if not batch:
                    break  # No more messages
                
                for msg in batch:
                    if msg.date is None:
                        continue
                        
                    msg_datetime = msg.date.replace(tzinfo=None)  # Remove timezone for comparison
                    
                    # Check if message is within date range
                    is_after_start = True if start_date is None else msg_datetime >= start_date
                    is_before_end = True if end_date is None else msg_datetime <= end_date
                    
                    if is_after_start and is_before_end and msg.message:
                        messages.append(msg)
                        total_collected += 1
                        if total_collected >= limit:
                            break
                
                if len(batch) < batch_size:
                    break  # No more messages to fetch
                    
                offset_id = batch[-1].id
        else:
            # Simple case: just fetch the requested number of messages
            messages = await client.get_messages(chat_id, limit=limit)
        
        # Sort messages by date (oldest first)
        messages.sort(key=lambda msg: msg.date)
        
        # Extract message text with sender information
        chat_text = extract_message_text(messages)
        
        print(f"Fetched {len(messages)} messages" + 
              (f" from {start_date} to {end_date}" if use_date_range else ""))
        
        return chat_text

def list_available_tts_models():
    """
    Lists available TTS models in the Coqui TTS library.
    Useful for finding models that might support different languages.
    """
    try:
        print("Available TTS models:")
        models = TTS().list_models()
        
        # Try to find models that might support Russian
        multilingual_models = [m for m in models if "multi" in m.lower()]
        print("\nMultilingual models that might support Russian:")
        for m in multilingual_models:
            print(f"- {m}")
        
        return models
    except Exception as e:
        print(f"Error listing TTS models: {e}")
        return []

# ---------------------------
# Main Routine
# ---------------------------

async def main():
    # Print a clear header for better visibility in Docker logs
    print("\n" + "*"*70)
    print("*" + " "*26 + "TELEGRAM BOT STARTUP" + " "*26 + "*")
    print("*"*70 + "\n")
    
    # Verify environment variables
    if not all([API_ID, API_HASH, PHONE_NUMBER]):
        print("ERROR: Missing Telegram credentials in environment variables!")
        print("Please check your .env file contains:")
        print("  TELEGRAM_API_ID")
        print("  TELEGRAM_API_HASH")
        print("  TELEGRAM_PHONE_NUMBER")
        sys.exit(1)

    # Initialize the Telegram client
    print(f"Initializing Telegram client with phone: {PHONE_NUMBER}")
    print(f"Session will be stored at: {SESSION_NAME}")
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

    # Connect and handle authentication explicitly
    print("\nConnecting to Telegram...")
    await client.connect()
    
    # Check if already authenticated
    if not await client.is_user_authorized():
        print("\n" + "="*50)
        print("AUTHENTICATION REQUIRED")
        print("="*50)
        print("You need to authenticate with Telegram.")
        print("A verification code will be sent to your Telegram account.")
        print("="*50 + "\n")
        
        try:
            # Request verification code
            print("Requesting verification code...")
            await client.send_code_request(PHONE_NUMBER)
            
            # Make the input prompt very visible
            print("\n" + "!"*50)
            print("! PLEASE CHECK YOUR TELEGRAM APP FOR THE VERIFICATION CODE !")
            print("!"*50 + "\n")
            
            # Use a large timeout to make sure we don't miss user input
            code = input("Enter the code you received: ")
            print(f"Received code: {code}")
            
            # Sign in with the code
            print("Signing in with provided code...")
            await client.sign_in(PHONE_NUMBER, code)
            print("\nSuccessfully authenticated with Telegram!")
        except Exception as e:
            print(f"\nError during authentication: {e}")
            print("Please try running the container again.")
            await client.disconnect()
            sys.exit(1)
    else:
        print("Already authenticated with Telegram!")

    print("\nStarting bot operation...\n")

    # List available TTS models (helpful for debugging)
    list_available_tts_models()
    
    print("Retrieving chat history...")
    chat_content = await fetch_chat_history(client)
    
    if not chat_content:
        print("No text messages found in the selected chat.")
        await client.disconnect()
        return
    
    # Handle forum topics differently than regular chats
    if isinstance(chat_content, dict):  # It's a forum with topics
        print("Processing summaries for each topic...")
        topic_summaries = {}
        
        # For each topic, generate a separate summary
        for topic_name, messages_text in chat_content.items():
            # Detect language for each topic
            language = detect_language(messages_text)
            print(f"Topic '{topic_name}' detected language: {language}")
            
            # Generate summary for this topic
            print(f"Generating summary for topic '{topic_name}'...")
            topic_summary = generate_summary(messages_text, language, topic_name)
            topic_summaries[topic_name] = topic_summary
        
        # Create a combined summary across all topics
        print("Generating consolidated podcast script from all topics...")
        
        # Use the language from the first topic as the primary language
        primary_language = detect_language(list(chat_content.values())[0])
        summary = generate_multi_topic_summary(topic_summaries, primary_language)
        
        print("Consolidated summary generated")
    else:  # Regular chat without topics
        # Detect the language of the chat
        language = detect_language(chat_content)
        print(f"Detected language: {language}")
        
        print("Generating podcast-style summary...")
        summary = generate_summary(chat_content, language)
        print("Summary generated")
    
    # Save the summary to a file for reference
    summary_file = os.path.join(OUTPUT_DIR, "podcast_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved to {summary_file}")
    
    # Determine language for TTS (use the detected language from earlier steps)
    language = primary_language if isinstance(chat_content, dict) else language
    
    print(f"Synthesizing speech for the podcast using TTS (Language: {language})...")
    podcast_file = synthesize_speech_with_coqui(summary, language)
    
    if not podcast_file:
        print("Failed to generate podcast audio file.")
        await client.disconnect()
        return
        
    print(f"Podcast audio saved to: {podcast_file}")
    
    # Optionally, send the podcast file back to yourself on Telegram:
    try:
        await client.send_file('me', podcast_file, caption=f"Your podcast summary ({language})")
        print(f"Podcast file sent to your Saved Messages.")
    except Exception as e:
        print(f"Error sending file: {e}")
    
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
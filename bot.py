import os
import asyncio
import datetime
from dotenv import load_dotenv
from telethon import TelegramClient
from openai import OpenAI
from TTS.api import TTS  # Coqui TTS
import langdetect  # For language detection
from gtts import gTTS  # Google Text-to-Speech as a fallback

# Load environment variables from .env file
load_dotenv()

# ---------------------------
# Configuration and Setup
# ---------------------------

# Telegram configuration (set these as environment variables or replace directly)
API_ID = os.environ.get("TELEGRAM_API_ID")         # e.g., "123456"
API_HASH = os.environ.get("TELEGRAM_API_HASH")       # e.g., "abcdef123456..."
PHONE_NUMBER = os.environ.get("TELEGRAM_PHONE_NUMBER")  # e.g., "+1234567890"
SESSION_NAME = "telegram_session"

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

def generate_summary(chat_text: str, language: str = "en") -> str:
    """
    Generates a podcast-style summary using OpenAI's GPT-3.5-turbo.
    The prompt instructs the model to produce a script that would last 5-10 minutes.
    Includes information about participants and their main points/theses.
    """
    # Adjust system prompt based on detected language
    if language == "ru":
        system_prompt = (
            "Вы - эксперт по написанию подкастов. Обобщите следующую чат-беседу "
            "в четкий, увлекательный сценарий подкаста, который будет длиться около 5-10 минут при "
            "чтении вслух. Сценарий должен быть разговорным и естественным. "
            "Обязательно включите следующие детали:\n"
            "1. Имена участников чата (кто отправлял сообщения)\n"
            "2. Кто участвовал в обсуждении\n"
            "3. Основные тезисы и ключевые точки дискуссии от каждого участника\n"
            "Пожалуйста, создайте сценарий на русском языке."
        )
    else:
        system_prompt = (
            "You are an expert podcast script writer. Summarize the following chat conversation "
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

def synthesize_speech_with_coqui(text: str, language: str = "en", output_filename: str = "podcast_summary.wav") -> str:
    """
    Converts the given text to natural-sounding speech using Coqui TTS.
    This uses a pre-trained model that can be run locally without cloud credentials.
    Falls back to Google TTS for Russian if needed.
    """
    try:
        # For Russian, try Google TTS first as it has reliable Russian support
        if language == "ru":
            try:
                print("Using Google TTS for Russian language")
                # Google TTS uses MP3 format
                mp3_output = output_filename.replace(".wav", ".mp3")
                gtts = gTTS(text=text, lang='ru', slow=False)
                gtts.save(mp3_output)
                print(f"Speech synthesized using Google TTS and saved to {mp3_output}")
                return mp3_output
            except Exception as e:
                print(f"Google TTS failed: {e}")
                print("Falling back to Coqui TTS models...")
                # Continue with Coqui TTS fallbacks
        
        # Original Coqui TTS logic
        # Select appropriate TTS model based on language
        if language == "ru":
            # Try different approaches for Russian language
            try:
                # First attempt with a multi-language model that might support Russian
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
                print("Using multilingual XTTS model for Russian")
            except Exception as e:
                print(f"Couldn't load multilingual XTTS model: {e}")
                # Fall back to YourTTS which has broader language support
                try:
                    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True, gpu=False)
                    print("Using YourTTS model for Russian")
                except Exception as e:
                    print(f"Couldn't load YourTTS model: {e}")
                    # Ultimate fallback to English model
                    print("Falling back to English model for Russian text")
                    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
        else:
            # Default English model
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
        
        # Convert text to speech and save as a WAV file.
        tts.tts_to_file(text=text, file_path=output_filename)
        return output_filename
    except Exception as e:
        print(f"Error synthesizing speech with Coqui TTS: {e}")
        print("Trying Google TTS as a final fallback...")
        try:
            # Use Google TTS as a final fallback
            lang_code = 'ru' if language == 'ru' else 'en'
            mp3_output = output_filename.replace(".wav", ".mp3")
            gtts = gTTS(text=text, lang=lang_code, slow=False)
            gtts.save(mp3_output)
            print(f"Speech synthesized using Google TTS fallback and saved to {mp3_output}")
            return mp3_output
        except Exception as e2:
            print(f"Google TTS fallback also failed: {e2}")
            print("Unable to synthesize speech. Returning empty file path.")
            return ""

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

async def fetch_chat_history(client: TelegramClient) -> str:
    """
    Retrieves chat history from a selected chat.
    Lists available chats and allows user to specify message limit and date range.
    """
    dialogs = await client.get_dialogs()
    print("Available chats:")
    for i, dialog in enumerate(dialogs):
        print(f"{i}: {dialog.name}")

    selected_index = int(input("Enter the index of the chat to summarize: "))
    selected_dialog = dialogs[selected_index]
    chat_id = selected_dialog.id
    
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
    
    # Fetch messages with the specified parameters
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
    
    chat_text = "\n".join(chat_lines)
    
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
    # Initialize and start the Telegram client as a userbot.
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start(phone=PHONE_NUMBER)
    
    # List available TTS models (helpful for debugging)
    list_available_tts_models()
    
    print("Retrieving chat history...")
    chat_text = await fetch_chat_history(client)
    
    if not chat_text.strip():
        print("No text messages found in the selected chat.")
        await client.disconnect()
        return
    
    # Detect the language of the chat
    language = detect_language(chat_text)
    print(f"Detected language: {language}")
    
    print("Generating podcast-style summary...")
    summary = generate_summary(chat_text, language)
    print("Summary generated:\n", summary)
    
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
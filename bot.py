import os
import asyncio
import datetime
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import Channel, Message
from telethon.tl.functions.channels import GetForumTopicsRequest
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
    # Initialize and start the Telegram client as a userbot.
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start(phone=PHONE_NUMBER)
    
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
    with open("podcast_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print("Summary saved to podcast_summary.txt")
    
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
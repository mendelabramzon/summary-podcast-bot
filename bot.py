import os
import asyncio
import datetime
from dotenv import load_dotenv
from telethon import TelegramClient
from openai import OpenAI
from TTS.api import TTS  # Coqui TTS

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

def generate_summary(chat_text: str) -> str:
    """
    Generates a podcast-style summary using OpenAI's GPT-3.5-turbo.
    The prompt instructs the model to produce a script that would last 5-10 minutes.
    """
    system_prompt = (
        "You are an expert podcast script writer. Summarize the following chat conversation "
        "into a clear, engaging podcast script that would last about 5-10 minutes when read aloud. "
        "The script should be conversational and natural."
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

def synthesize_speech_with_coqui(text: str, output_filename: str = "podcast_summary.wav") -> str:
    """
    Converts the given text to natural-sounding speech using Coqui TTS.
    This uses a pre-trained model that can be run locally without cloud credentials.
    """
    # Initialize the TTS model; this example uses a pre-trained model for English.
    # You can choose different models available in the TTS package.
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
    
    # Convert text to speech and save as a WAV file.
    tts.tts_to_file(text=text, file_path=output_filename)
    return output_filename

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
    
    # Extract message text
    chat_text = "\n".join(msg.message for msg in messages if msg.message)
    
    print(f"Fetched {len(messages)} messages" + 
          (f" from {start_date} to {end_date}" if use_date_range else ""))
    
    return chat_text

# ---------------------------
# Main Routine
# ---------------------------

async def main():
    # Initialize and start the Telegram client as a userbot.
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start(phone=PHONE_NUMBER)
    
    print("Retrieving chat history...")
    chat_text = await fetch_chat_history(client)
    
    if not chat_text.strip():
        print("No text messages found in the selected chat.")
        await client.disconnect()
        return

    print("Generating podcast-style summary...")
    summary = generate_summary(chat_text)
    print("Summary generated:\n", summary)
    
    print("Synthesizing speech for the podcast using Coqui TTS...")
    podcast_file = synthesize_speech_with_coqui(summary)
    print(f"Podcast audio saved to: {podcast_file}")
    
    # Optionally, send the podcast file back to yourself on Telegram:
    await client.send_file('me', podcast_file, caption="Your podcast summary")
    
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
# Telegram Chat to Podcast Bot

This bot fetches chat history from your Telegram conversations, generates a podcast-style summary using OpenAI's GPT, and converts it to speech using Coqui TTS.

## Features

- Fetches messages from any Telegram chat with customizable message limit
- Filters messages by date range for targeted summaries
- Creates a natural-sounding podcast script summarizing the conversation
- Converts the summary to speech with high-quality text-to-speech
- Sends the audio file back to you on Telegram

## Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12 (required for coqui-tts)
- Telegram API credentials
- OpenAI API key

### Conda Environment Setup (Recommended)

The coqui-tts package requires Python 3.10-3.12. Since you're running Python 3.13, it's recommended to create a conda environment:

1. Install Miniconda or Anaconda if you haven't already.

2. Create a new conda environment with Python 3.10:
   ```bash
   conda create -n telegram-podcast python=3.10
   conda activate telegram-podcast
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Standard Installation

If you already have Python 3.10-3.12:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telegram-chat-podcast.git
   cd telegram-chat-podcast
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   Create a `.env` file in the project root with the following:
   ```
   TELEGRAM_API_ID=your_api_id
   TELEGRAM_API_HASH=your_api_hash
   TELEGRAM_PHONE_NUMBER=your_phone_number_with_country_code
   OPENAI_API_KEY=your_openai_api_key
   ```

   Or set them in your environment:
   ```bash
   export TELEGRAM_API_ID="your_api_id"
   export TELEGRAM_API_HASH="your_api_hash"
   export TELEGRAM_PHONE_NUMBER="your_phone_number"
   export OPENAI_API_KEY="your_openai_key"
   ```

### How to Get Telegram API Credentials

1. Visit [my.telegram.org](https://my.telegram.org/)
2. Log in with your phone number
3. Go to "API development tools"
4. Create a new application (any name will work)
5. Note your API ID and API Hash

## Usage

1. Run the bot:
   ```bash
   python bot.py
   ```

2. First-time setup:
   - You'll be prompted to verify your phone number with a code sent via Telegram
   - This creates a session file for future authentication

3. Bot workflow:
   - The bot will display a list of your chats
   - Enter the index number of the chat you want to summarize
   - Specify how many messages to fetch (defaults to 1000)
   - Choose whether to filter by date range:
     - If yes, enter start and end dates (flexible format support)
     - Leave start date empty to fetch from the earliest messages
     - Leave end date empty to use today's date as the end
   - The bot will fetch messages based on your criteria
   - It will generate a podcast-style summary using GPT-3.5
   - The summary will be converted to speech
   - The audio file will be saved as "podcast_summary.wav"
   - The bot will also send the audio back to you on Telegram

## Customization

You can modify these parameters in the code:
- `max_tokens` in `generate_summary()` to control summary length
- TTS model in `synthesize_speech_with_coqui()` for different voices
- `DEFAULT_MESSAGE_LIMIT` to change the default number of messages

## License

[MIT License](LICENSE)

## Disclaimer

This bot uses your Telegram account as a userbot. Make sure you comply with Telegram's Terms of Service when using this tool. 